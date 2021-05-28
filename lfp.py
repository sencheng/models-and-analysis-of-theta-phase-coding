import sys
import h5py
import json
import numpy as np
from scipy import signal
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from data_analysis.general import Base
from data_analysis.spikes_basics import load_spikes


class LFP(Base):
    """Class for loading and working with LFP data.

    Args:
        super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is defined
            as belonging to the super-group, it will be shared across sub-groups.
        group_name (string): Name of the low-level sub-group used for pickles and figures.
        child_name (string): Name of the instance used for pickles and figures.
        filter_order (int): Order of the Butterworth filter used for filtering the LFP.
        bandpass_frequencies (tuple(float)): (lower bound, higher bound) frequencies for the band pass filter.
        save_figures (bool): Whether to save the figures.

    Attributes:
        sampling_rate (int): Sampling rate (Hz).
        channels (list(int)): Channels analyzed.
        num_channels (int): Number of channels analyzed.
        times (np.array): Time steps (s).
        raw_lfp (np.array): Raw LFP signal loaded from the binary file.
        theta_filtered_lfp (np.array): Filtered LFP signal.
        analytic_signal (np.array): Hilbert transform of the filtered LFP.
        cycle_boundaries (list(list(float))): For each channel, time indices of detected peaks in the LFP.
        inst_phase (np.array): Instantaneous phase of the filtered LFP signal (rad).
        inst_frequency (np.array): Instantaneous frequency of the filtered LFP signal (Hz).
        inst_amplitude (np.array): Instantaneous amplitude of the Hilbert transform.
        significant_amplitude_cut_offs (list(float)): Instantaneous amplitude significance cut-offs for each channel.
        significant (np.array): For each time step, for each channel, whether the theta oscillation is significant.
        significant_cycles (np.array): For each cycle boundary, whether the theta oscillation is
            significant in the cycle.

    """
    belongs_to_super_group = True

    def __init__(self, super_group_name, group_name, child_name, filter_order=3, bandpass_frequencies=(4, 12),
                 save_figures=False, figure_format="png", figures_path="figures", pickle_results=True,
                 pickles_path="pickles"):

        super().__init__(super_group_name, group_name, child_name, save_figures, figure_format, figures_path,
                         pickle_results, pickles_path)

        self.filter_order = filter_order
        self.bandpass_frequencies = bandpass_frequencies

        self.sampling_rate = None
        self.times = None
        self.channels = None
        self.num_channels = None
        self.raw_lfp = None
        self.theta_filtered_lfp = None
        self.filtered_lfp = None  # for broader band filtering
        self.analytic_signal = None
        self.significant = None
        self.significant_amplitude_cut_offs = []
        self.inst_phase = None
        self.inst_frequency = None
        self.inst_amplitude = None
        # self.inst_significant = None
        self.cycle_boundaries = None  # roughly corresponding to the peaks
        self.cycle_peaks = None
        self.cycle_troughs = None
        self.significant_cycles = None

    def load_lfp(self, data_path, dataset, session_set, session, channels=None):
        """
        Load the data.

        Args:
            data_path (string): Path to the data.
            dataset (string): Name of the dataset, e.g., hc-3.
            session_set (string): Name of the set of sessions that get lumped together, e.g., ec013.28.
            session (string): Name of the session, e.g., ec013.412.
            channels (list(int)): List of LFP channels to keep, or None to keep all.
        """
        path = f"{data_path}/{dataset}/{session_set}/{session}/{session}"
        xml_tree = ET.parse(f'{path}.xml')
        num_channels = int(xml_tree.find('acquisitionSystem/nChannels').text)
        print(f'number of channels: {num_channels}')
        sampling_rate = int(xml_tree.find('fieldPotentials/lfpSamplingRate').text)
        self.sampling_rate = sampling_rate
        print(f'LFP sampling rate: {sampling_rate}')

        if dataset == 'hc-11':
            with h5py.File(f'{path}_sessInfo.mat', 'r') as f:
                maze_epoch = np.array(f['sessInfo']['Epochs']['MazeEpoch'])
            first_byte = int(maze_epoch[0]*sampling_rate)*num_channels*2
            num_bytes = int((maze_epoch[1] - maze_epoch[0])*sampling_rate)*num_channels*2
            serialized_lfp = np.fromfile(f'{path}.eeg', dtype='int16', offset=first_byte, count=num_bytes)
            self.times = np.arange(int(len(serialized_lfp) / num_channels)) / sampling_rate + maze_epoch[0]
        else:
            serialized_lfp = np.fromfile(f'{path}.eeg', dtype='int16')
            if len(serialized_lfp) % num_channels != 0:
                sys.exit('number of samples is not a multiple of the number of channels')
            self.times = np.arange(int(len(serialized_lfp) / num_channels)) / sampling_rate
        print(f'LFP data from {self.times[0]} to {self.times[-1]} s')

        if channels is None:
            self.channels = list(range(num_channels))
        else:
            self.channels = channels
        self.num_channels = len(self.channels)

        self.raw_lfp = serialized_lfp.reshape((int(len(serialized_lfp) / num_channels)), num_channels)[:, self.channels]

    def generate_constant_theta(self, duration, sampling_rate=1250, frequency=8, num_channels=1):
        self.sampling_rate = sampling_rate
        self.channels = list(range(num_channels))
        self.num_channels = num_channels
        self.times = np.arange(0, duration, 1/sampling_rate)
        theta = np.cos(2*np.pi*frequency*self.times)[np.newaxis].T
        self.raw_lfp = np.hstack([theta for _ in range(num_channels)])

    def finish_initialization(self):
        self.significant = np.full(self.raw_lfp.shape, False).astype(bool)
        # self.inst_significant = np.full(self.raw_lfp.shape, True).astype(bool)

        self.cycle_boundaries = [[] for _ in range(self.num_channels)]
        self.significant_cycles = [[] for _ in range(self.num_channels)]

        self.theta_filtered_lfp = self.butterworth_filter(self.filter_order, self.bandpass_frequencies)
        self.analytic_signal = np.empty(self.raw_lfp.shape).astype(np.csingle)
        for channel_num in range(self.num_channels):
            self.analytic_signal[:, channel_num] = self.padded_hilbert(-self.theta_filtered_lfp[:, channel_num])
        self.inst_amplitude = np.absolute(self.analytic_signal).astype(np.float16)

    @staticmethod
    def padded_hilbert(input_signal):
        """Calculates the Hilbert transform of a signal with zero padding to the closest power of two,
        which makes it much more efficient.

        Args:
            input_signal (np.array): 1D Vector with the input signal.

        Returns:
            (np.array): Hilbert transform.
        """
        padding = np.zeros(int(2 ** np.ceil(np.log2(len(input_signal)))) - len(input_signal))
        to_hilbert = np.hstack((input_signal, padding))
        result = signal.hilbert(to_hilbert)
        return result[0:len(input_signal)]

    def butterworth_filter(self, filter_order=3, bandpass_frequencies=(4, 12)):
        """Filter the LFP with a Butterworth bandpass filter.

        Args:
            filter_order (int): Order of the filter.
            bandpass_frequencies (tuple(float)): Lower and upper bound for the bandpass filter (Hz).
        """
        print('filtering...')
        filtered_lfp = np.empty(self.raw_lfp.shape)
        filter_params = signal.butter(filter_order, bandpass_frequencies, btype='bandpass', output='ba',
                                      fs=self.sampling_rate)
        for channel_num in range(self.num_channels):
            filtered_lfp[:, channel_num] = signal.filtfilt(filter_params[0], filter_params[1],
                                                           self.raw_lfp[:, channel_num])
        return filtered_lfp

    def phase_from_hilbert_transform(self, peak_height=350):
        """Calculate the instantaneous phase from the Hilbert's transform.
        """
        print('estimating phase from the Hilbert transform...')
        self.inst_phase = np.angle(self.analytic_signal) / np.pi * 180 + 180

    def phase_from_peaks(self):
        """Calculate the phase of theta as the distance between successive peaks of the LFP.
        """
        print('estimating phase from peaks...')
        self.analytic_signal = None  # won't need it
        self.cycle_peaks = [[] for _ in range(self.num_channels)]

        for channel_num in range(self.num_channels):
            self.cycle_peaks[channel_num] = signal.find_peaks(self.theta_filtered_lfp[:, channel_num])[0]

        next_boundary = np.full((len(self.times), self.num_channels), np.nan)
        previous_boundary = np.full((len(self.times), self.num_channels), np.nan)
        for channel_num in range(self.num_channels):
            for previous_peak, next_peak in zip(self.cycle_peaks[channel_num][:-1],
                                                self.cycle_peaks[channel_num][1:]):
                previous_boundary[previous_peak:next_peak, channel_num] = previous_peak
                next_boundary[previous_peak:next_peak, channel_num] = next_peak
                self.cycle_boundaries[channel_num].append((previous_peak, next_peak))

        time_indices = np.arange(len(self.times)).reshape((len(self.times), 1))
        self.inst_phase = (time_indices - previous_boundary) / (next_boundary - previous_boundary) * 360

    def phase_from_waveform(self, bandpass_frequencies=(1, 60), filter_order=3, min_peak_distance=0.08, prominence=500):
        # bandpass filter
        self.filtered_lfp = self.butterworth_filter(filter_order, bandpass_frequencies)

        # find peaks and troughs
        min_distance = min_peak_distance * self.sampling_rate
        self.cycle_peaks = [[] for _ in range(self.num_channels)]
        self.cycle_troughs = [[] for _ in range(self.num_channels)]
        self.inst_phase = np.full(self.raw_lfp.shape, np.nan)
        for channel_num in range(self.num_channels):
            self.cycle_peaks[channel_num] = signal.find_peaks(self.filtered_lfp[:, channel_num], height=0,
                                                              distance=min_distance, prominence=prominence)[0]
            self.cycle_troughs[channel_num] = signal.find_peaks(-self.filtered_lfp[:, channel_num], height=0,
                                                                distance=min_distance, prominence=prominence)[0]

            # interpolate phases
            for first_peak, second_peak in zip(self.cycle_peaks[channel_num][:-1], self.cycle_peaks[channel_num][1:]):
                troughs = self.cycle_troughs[channel_num][(self.cycle_troughs[channel_num] > first_peak) &
                                                          (self.cycle_troughs[channel_num] < second_peak)]
                if len(troughs) == 1:
                    self.inst_phase[first_peak:troughs[0]+1, channel_num] = np.linspace(0, 180, troughs[0]+1-first_peak)
                    self.inst_phase[troughs[0]:second_peak+1, channel_num] = np.linspace(180, 360, second_peak+1-troughs[0])
                    self.cycle_boundaries[channel_num].append((first_peak, second_peak))

    def find_significant_theta(self, amplitude_percentile=97, plot_steps=False, plot_histogram=True, bins=50,
                               high_pass_frequency=1):
        """Find the significance cut-off value for the instantaneous amplitude of the signal through shuffling.

        Args:
            amplitude_percentile (float): Percentile of the shuffled distribution.
            plot_steps (bool): Plot the shuffled and filtered LFP and its instantaneous amplitude.
            plot_histogram (bool): Plot histogram of shuffled instantaneous amplitudes.
            bins (int): Number of bins for the histogram.
            high_pass_frequency (float): Cut-off frequency for the high pass filter that is applied before the shuffling.
        """
        print('finding significance cut-offs via shuffling...')

        for channel_num in range(self.num_channels):
            high_pass = signal.butter(self.filter_order, high_pass_frequency, btype='highpass', output='ba',
                                      fs=self.sampling_rate)
            high_passed_lfp = signal.filtfilt(high_pass[0], high_pass[1], self.raw_lfp[:, channel_num])
            shuffled_lfp = high_passed_lfp[np.random.permutation(self.raw_lfp.shape[0])]
            filter_params = signal.butter(self.filter_order, self.bandpass_frequencies, btype='bandpass', output='ba',
                                          fs=self.sampling_rate)
            filtered_shuffled_lfp = signal.filtfilt(filter_params[0], filter_params[1], shuffled_lfp)
            instantaneous_amplitude = np.absolute(self.padded_hilbert(-filtered_shuffled_lfp))
            significant_amplitude_cut_off = np.percentile(instantaneous_amplitude, amplitude_percentile)
            self.significant_amplitude_cut_offs.append(significant_amplitude_cut_off)
            # self.inst_significant[:, channel_num] = self.inst_amplitude[:, channel_num] >= significant_amplitude_cut_off

            if plot_histogram:
                fig, ax = plt.subplots()
                ax.hist(instantaneous_amplitude, bins=bins, density=True)
                ax.axvline(self.significant_amplitude_cut_offs[channel_num], linestyle='dotted', color='r',
                           label=f'{amplitude_percentile} percentile')
                ax.set_xlabel('Magnitude of the Hilbert transform (?)')
                ax.set_ylabel('Normalized counts')
                ax.legend()
                self.maybe_save_fig(fig, f"shuffled_signal_amplitude_histogram_{self.channels[channel_num]}")

            if plot_steps:
                fig, ax = plt.subplots(2, sharex='col')
                ax[0].plot(self.times, self.raw_lfp[:, channel_num], label='raw')
                ax[0].plot(self.times, high_passed_lfp, label='high passed')
                ax[0].set_ylabel('LFP (?)')
                ax[0].legend(loc='lower right')
                ax[1].plot(self.times, shuffled_lfp, label='shuffled')
                ax[1].plot(self.times, filtered_shuffled_lfp, label='filtered')
                ax[1].plot(self.times, instantaneous_amplitude, label='amplitude')
                ax[1].set_xlabel('Time (s)')
                ax[1].set_ylabel('LFP (?)')
                ax[1].legend(loc='lower right')

    def calculate_phase_locking(self, data_path, dataset, session_set, session, discarded_intervals, region='CA1',
                                num_bins=36, sigma=10, plot=False, min_firing_to_zero=False):
        """Calculate a histogram of spike count vs theta phase for periods of significant theta oscillation and
        set phase 0 as the phase of minimum spiking.

        Args:
            data_path (string): Path to the data.
            dataset (string): Name of the dataset, e.g., hc-3.
            session_set (string): Name of the set of sessions that get lumped together, e.g., ec013.28.
            session (string): Name of the session, e.g., ec013.412.
            region (string): Brain region in which the electrodes were placed (as per the metadata table).
            num_bins (int): Number of bins in the histogram.
            sigma (float): Standard deviation used for smoothing the histogram with a Gaussian filter (deg).
            plot (bool): Whether to plot the histogram.
            min_firing_to_zero (bool): Set phase 0 as the phase of minimum firing.
        """
        spikes = load_spikes(data_path, dataset, session_set, session, discarded_intervals, region)
        phases = []
        for electrode_cluster_pair, pair_spikes_times in zip(spikes.electrode_cluster_pairs, spikes.spike_times):
            electrode_index = spikes.electrodes.index(electrode_cluster_pair[0])
            for spike_time in pair_spikes_times:
                phase, significant = self.at_time(spike_time, channel_index=electrode_index, return_phase=True,
                                                  return_significance=True)
                if significant:
                    phases.append(phase)

        hist, bin_edges = np.histogram(phases, bins=num_bins, range=(0, 360))
        smooth_hist = gaussian_filter1d(hist, sigma/(360/num_bins))
        x = np.mean(np.vstack((bin_edges[:-1], bin_edges[1:])), axis=0)
        phase_0 = x[np.argmin(smooth_hist)]

        if min_firing_to_zero:
            self.inst_phase = (self.inst_phase - phase_0) % 360

        if plot:
            fig, ax = plt.subplots()
            ax.bar(x, hist, width=360/num_bins)
            ax.plot(x, smooth_hist, color='C1')
            ax.plot(phase_0, np.min(smooth_hist), '*', color='C3')
            ax.set_xlabel("Phase (deg)")
            ax.set_ylabel("Spike count")
            self.maybe_save_fig(fig, "phase_locking")

    def find_cycle_boundaries(self, peak_height=320):
        """Define cycle boundaries based on detecting peaks in the instantaneous phase.

        Args:
            peak_height (float): Minimum height of the peak in the instantaneous phase change in order to mark a new
                cycle.
        """
        for channel_num in range(self.num_channels):
            peaks = signal.find_peaks(self.inst_phase[:, channel_num], height=peak_height)[0]
            self.cycle_boundaries[channel_num] = np.vstack((peaks[:-1], peaks[1:])).T

    def find_significant_cycles(self):
        """Fill in a variable indicating whether each time step for each channel belongs to a cycle that is entirely
        statistically significant (or valid).
        """
        significant_ratios = []
        for channel_num in range(self.num_channels):
            for left_cycle_boundary, right_cycle_boundary in self.cycle_boundaries[channel_num]:
                if (self.inst_amplitude[left_cycle_boundary:right_cycle_boundary, channel_num] <
                        self.significant_amplitude_cut_offs[channel_num]).any():
                    self.significant_cycles[channel_num].append(False)
                else:
                    self.significant[left_cycle_boundary:right_cycle_boundary, channel_num] = True
                    self.significant_cycles[channel_num].append(True)
            significant_ratios.append(np.sum(self.significant_cycles[channel_num]) /
                                      len(self.significant_cycles[channel_num]))
            print(f'{significant_ratios[-1]*100:.2f}%'
                  f' cycles significant in channel {channel_num}')
        self.maybe_pickle_results(significant_ratios, "significant_ratios")

    def at_time(self, time, channel_index, return_phase=False, return_significance=False):
        """Returns the phase corresponding to the time point closes to time for some given channel index.

        Args:
            time (float): Time (s).
            channel_index (int): LFP channel index.
            return_phase (bool): Return theta phase.
            return_significance (bool): Return significance of the cycle.

        Returns:
            (float): Instantaneous phase (rad).
        """
        time_index = min(int(round((time-self.times[0]) * self.sampling_rate)), self.significant.shape[0] - 1)
        returns = []
        if return_phase:
            returns.append(self.inst_phase[time_index, channel_index])
        if return_significance:
            returns.append(self.significant[time_index, channel_index])
        # if return_inst_significance:
        #     returns.append(self.inst_significant[time_index, channel_index])
        return returns

    def comp_instantaneous_frequency(self):
        """Compute instantaneous frequency of the LFP as the first difference of the signal.
        """
        self.inst_frequency = np.full(self.raw_lfp.shape, np.nan)
        instantaneous_phase_in_radians = self.inst_phase / 180 * np.pi - np.pi
        for channel_num in range(self.num_channels):
            not_nan = ~np.isnan(instantaneous_phase_in_radians[:, channel_num])
            self.inst_frequency[not_nan, channel_num] = np.append(np.diff(np.unwrap(
                instantaneous_phase_in_radians[not_nan, channel_num])) / (2 * np.pi) * self.sampling_rate, np.nan)

    def plot(self, channels, time_interval, plot_phase=True, plot_frequency=True):
        """Plot the LFP.

        Args:
            channels (list(int)): Channels to plot.
            time_interval (list(float)): Upper and lower bounds of the time interval to plot.
            plot_phase (bool): Whether or not to plot the instantaneous phase.
            plot_frequency (bool): Whether or not to plot the instantaneous frequency.
        """
        fig, ax = plt.subplots(1 + plot_phase + plot_frequency, sharex='col')

        first_index = np.searchsorted(self.times, time_interval[0])
        last_index = np.searchsorted(self.times, time_interval[1])
        time_slice = slice(first_index, last_index)

        for channel in channels:
            channel_num = self.channels.index(channel)
            ax[0].plot(self.times[time_slice], self.raw_lfp[time_slice, channel_num], label=f'channel: {channel}, raw')
            ax[0].plot(self.times[time_slice], self.theta_filtered_lfp[time_slice, channel_num],
                       label=f'channel: {channel}, theta band filtered')
            ax[0].plot(self.times[time_slice],
                       np.where(self.significant[time_slice, channel_num],
                                self.theta_filtered_lfp[time_slice, channel_num], np.nan),
                       label=f'channel: {channel}, theta band filtered (significant)')
            ax[0].plot(self.times[time_slice], self.inst_amplitude[time_slice, channel_num],
                       label=f'channel: {channel}, amplitude')
            ax[0].axhline(self.significant_amplitude_cut_offs[channel_num])

            cycle_boundaries = np.array(self.cycle_boundaries[channel_num])
            cycle_boundaries = cycle_boundaries[(cycle_boundaries[:, 0] >= first_index) &
                                                (cycle_boundaries[:, 1] < last_index)]

            if self.cycle_troughs is not None:  # used the waveform method for estimating phase
                ax[0].plot(self.times[time_slice], self.filtered_lfp[time_slice, channel_num],
                           label=f'channel: {channel}, broadband filtered')
                cycle_troughs = self.cycle_troughs[channel_num][
                    (self.cycle_troughs[channel_num] >= first_index)
                    & (self.cycle_troughs[channel_num] < last_index)]
                ax[0].plot(self.times[cycle_troughs], self.filtered_lfp[cycle_troughs, channel_num], '*', color='C7')
                cycle_peaks = self.cycle_peaks[channel_num][
                    (self.cycle_peaks[channel_num] >= first_index) & (self.cycle_peaks[channel_num] < last_index)]
                ax[0].plot(self.times[cycle_peaks], self.filtered_lfp[cycle_peaks, channel_num], '*k')
            else:
                ax[0].plot(self.times[cycle_boundaries[:, 0]], self.theta_filtered_lfp[cycle_boundaries[:, 0], channel_num], '*k')
                ax[0].plot(self.times[cycle_boundaries[:, 1]],
                           self.theta_filtered_lfp[cycle_boundaries[:, 1], channel_num], '*k')

            if plot_phase:
                ax[1].plot(self.times[time_slice], self.inst_phase[time_slice, channel_num], '.-',
                           label=f'channel: {channel}')
                ax[1].plot(self.times[cycle_boundaries[:, 0]], self.inst_phase[cycle_boundaries[:, 0], channel_num], '.k')
                ax[1].plot(self.times[cycle_boundaries[:, 1]], self.inst_phase[cycle_boundaries[:, 1], channel_num], '.k')

            if plot_frequency:
                ax[2].plot(self.times[time_slice], self.inst_frequency[time_slice, channel_num],
                           label=f'channel: {channel}')

        ax[0].legend(loc='lower left')
        ax[0].set_ylabel('LFP\n(?)')

        if plot_phase:
            ax[1].set_ylabel('Instantaneous\nphase (deg)')
            ax[1].legend(loc='lower left')

        if plot_frequency:
            ax[2].set_ylabel('Instantaneous\nfrequency (Hz)')
            ax[2].legend(loc='lower left')

        ax[-1].set_xlabel('Time (s)')
        fig.align_ylabels()
        plt.tight_layout()
        self.maybe_save_fig(fig, "lfp")

    def clean(self):
        """Delete memory-intensive variables which are not used outside of this instance.
        """
        self.raw_lfp = None
        self.theta_filtered_lfp = None
        self.analytic_signal = None
        self.inst_amplitude = None
        # self.inst_significant = None

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=True, pickles_path="", **kwargs):

        lfp = cls(super_group_name, group_name, child_name, parameters_dict['filter_order'],
                  parameters_dict['bandpass_frequencies'], save_figures=True, figure_format=figure_format,
                  figures_path=figures_path, pickle_results=pickle_results, pickles_path=pickles_path)

        lfp.load_lfp(kwargs['data_path'], parameters_dict['dataset'], parameters_dict['session_set'],
                     parameters_dict['session'], parameters_dict['lfp_channels'])
        lfp.finish_initialization()

        if parameters_dict['phase_from'] == 'peaks':
            lfp.phase_from_peaks()

        elif parameters_dict['phase_from'] == 'hilbert':
            lfp.phase_from_hilbert_transform()
            with open(f"sessions/{super_group_name}.json", 'r') as f:
                session_dict = json.load(f)
            if 'phase_shift' in session_dict:
                lfp.inst_phase = (lfp.inst_phase + session_dict['phase_shift']) % 360
            lfp.find_cycle_boundaries()

        elif parameters_dict['phase_from'] == 'waveform':
            lfp.phase_from_waveform()

        lfp.find_significant_theta(plot_histogram=False,
                                   amplitude_percentile=parameters_dict['significance_percentile'],
                                   high_pass_frequency=parameters_dict['pre_shuffling_high_pass_frequency'])
        lfp.find_significant_cycles()
        lfp.clean()

        return lfp
