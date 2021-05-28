import numpy as np
import matplotlib.pyplot as plt
from data_analysis.general import Base
from data_analysis.lfp import LFP
from data_analysis.spikes_basics import Spikes, SpikesBase
from data_analysis.tracking import Tracking
from data_analysis.firing_fields import FiringFields


class Decoder(Base):
    """Class for performing Bayesian decoding of place cell spiking data.
    Follows the method described by Davidson, Kloosterman and Wilson, 2009 (Simple Bayesian decoding with uniform prior
    over positions).

    Args:
        super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is defined
            as belonging to the super-group, it will be shared across sub-groups.
        group_name (string): Name of the low-level sub-group used for pickles and figures.
        child_name (string): Name of the instance used for pickles and figures.
        spikes (Spikes): Spikes instance.
        tracking (Tracking): Tracking instance.
        firing_fields (FiringFields): FiringFields instance.
        save_figures (bool): Whether to save the figures.

    Attributes:
        num_spatial_bins (int): Number of spatial bins (taken from the rate maps).
        decoded_times (np.array): Times corresponding to the middle of each decoded time bin (s).
        spike_counts (np.array): Spike counts for each cell and decoding time window.
        decoded_probabilities (np.array): Matrix of decoded probabilities.
        real_positions (np.array): Tracking positions at the middle of each decoded time bin (cm).
    """
    dependencies = (LFP, SpikesBase, Tracking, FiringFields)

    def __init__(self, super_group_name, group_name, child_name, spikes, tracking, firing_fields, save_figures=False,
                 figure_format="png", figures_path="figures", pickle_results=True, pickles_path="pickles"):

        super().__init__(super_group_name, group_name, child_name, save_figures, figure_format, figures_path,
                         pickle_results, pickles_path)

        self.spikes = spikes
        self.tracking = tracking
        self.firing_fields = firing_fields
        self.num_spatial_bins = firing_fields.num_bins

        self.decoded_times = None
        self.spike_counts = None
        self.decoded_probabilities = None
        self.phase_bin_size = None
        self.phase_step_size = None
        self.phase_bins_per_cycle = None
        self.real_positions = None
        self.central_positions = None
        self.speeds = None
        self.run_types = None

    def count_spikes(self, time_start, time_stop):
        """Counts the number of spikes for each cell during a given time interval.

        Args:
            time_start (float): Beginning of the interval (s).
            time_stop (float): End of the interval (s).

        Returns:
            (np.array): Spike counts.
        """
        spike_counts = np.empty(len(self.spikes.electrode_cluster_pairs))
        for pair_num, spike_times in enumerate(self.spikes.spike_times):
            spike_counts[pair_num] = np.searchsorted(spike_times, time_stop) - np.searchsorted(spike_times, time_start)
        return spike_counts

    def decode_phase_bins(self, lfp, phase_bin_size, phase_step_size, time_interval=None, min_spikes=1,
                          plot_decoded_probabilities=False):

        if time_interval is None:
            time_interval = [0, self.tracking.times[-1]]
        self.phase_bin_size = phase_bin_size
        self.phase_step_size = phase_step_size
        self.phase_bins_per_cycle = int((360 - self.phase_bin_size) / self.phase_step_size) + 1

        # find theta peaks within the time interval
        all_left_boundaries = lfp.times[lfp.cycle_boundaries[0][:, 0]]
        selected_cycle_indices = (time_interval[0] < all_left_boundaries) & (all_left_boundaries < time_interval[1])
        significant_cycles = np.array(lfp.significant_cycles[0])[selected_cycle_indices]

        # for each theta cycle, find: central position, running speed, run type, and start and end times of time bins
        central_positions = []
        speeds = []
        run_types = []
        start_times = []
        end_times = []

        for cycle_num, (cycle_boundaries, significant_cycle) in \
                enumerate(zip(lfp.cycle_boundaries[0][selected_cycle_indices], significant_cycles)):
            if significant_cycle:
                left_boundary = lfp.times[cycle_boundaries[0]]
                right_boundary = lfp.times[cycle_boundaries[1]]
                central_position, run_type, speed = \
                    self.tracking.at_time((left_boundary + right_boundary) / 2, d_runs=True, return_speed=True)

                if run_type >= 0 and speed >= 0:

                    left_peak_run_type = self.tracking.run_type[np.searchsorted(self.tracking.times, left_boundary)]
                    right_peak_run_type = self.tracking.run_type[np.searchsorted(self.tracking.times, right_boundary)]

                    if run_type == left_peak_run_type == right_peak_run_type:
                        central_positions.append(central_position)
                        run_types.append(run_type)
                        cycle_length = right_boundary - left_boundary
                        cycle_phase_bin_size = cycle_length / 360 * phase_bin_size
                        cycle_start_times = np.linspace(left_boundary, right_boundary - cycle_phase_bin_size,
                                                        self.phase_bins_per_cycle)
                        start_times.extend(cycle_start_times)
                        end_times.extend(cycle_start_times + cycle_phase_bin_size)
                        speeds.append(speed)

        self.central_positions = np.array(central_positions)
        self.speeds = np.array(speeds)
        self.run_types = np.array(run_types)
        start_times = np.array(start_times)
        end_times = np.array(end_times)

        # run the decoder
        self.decode(start_times, end_times, min_spikes)

        if plot_decoded_probabilities:
            fig, ax = plt.subplots()
            ax.matshow(self.decoded_probabilities, aspect="auto", origin="lower", cmap="hot",
                       extent=(-0.5, start_times.size-0.5,
                               self.firing_fields.positions[0] - self.firing_fields.bin_size / 2,
                               self.firing_fields.positions[-1] + self.firing_fields.bin_size / 2))
            phase_bins_per_cycle = len(np.arange(0, 360 - self.phase_bin_size, self.phase_step_size))
            for peak in np.arange(-0.5, len(start_times), phase_bins_per_cycle):
                ax.axvline(peak)
            ax.plot(np.arange(phase_bins_per_cycle/2, len(start_times), phase_bins_per_cycle),
                    self.central_positions + self.firing_fields.positions[0], 'C1')
            ax.xaxis.set_ticks_position("bottom")
            ax.set_xlabel("Phase bin")
            ax.set_ylabel("Position (cm)")

    def decode_time_bins(self, bin_size, step_size, time_interval=(0, None), min_spikes=1, plot=True,
                         plot_most_probable_positions=False, plot_theta_peaks=False, lfp=None, plot_spike_counts=False):
        """Decode on a temporal sliding window.

        Args:
            bin_size (float): Size of the time window (s).
            step_size (float): Stride/step for the sliding window (s).
            time_interval (tuple(float)): Time interval (s) within which to decode.
            min_spikes (int): Minimum number of spikes for running the decoder.
            plot (bool): Plot the real trajectory overlaid over the decoded probabilities.
            plot_most_probable_positions (bool): Plot the positions of maximum decoded probability.
            plot_theta_peaks (bool): Plot vertical lines indicating the peaks of theta.
            lfp (LFP): LFP instance.
            plot_spike_counts (bool): Plot spike counts.
        """
        if time_interval[1] is None:
            time_interval[1] = self.tracking.times[-1]

        start_times = np.arange(time_interval[0], time_interval[1] - bin_size, step_size)
        end_times = start_times + bin_size

        self.decode(start_times, end_times, min_spikes)

        if plot:
            self.plot_decoded_probabilities(step_size, plot_most_probable_positions, plot_theta_peaks, lfp)
        if plot_spike_counts:
            self.plot_spike_counts(bin_size)

    def decode(self, start_times, end_times, min_spikes):
        """Runs the Bayesian decoding algorithm on a set of time bins defined by start_times and end_times.

        Args:
            start_times (np.array): List of start times for the time intervals.
            end_times (np.array): List of end times for the time intervals.
            min_spikes (int): Minimum number of spikes for running the decoder.
        """
        self.decoded_times = (start_times + end_times) / 2
        self.decoded_probabilities = np.full((self.num_spatial_bins, start_times.size), np.nan)
        self.real_positions = np.full(start_times.size, np.nan)
        self.spike_counts = np.full((len(self.spikes.electrode_cluster_pairs), start_times.size), np.nan)
        firing_rates = np.nan_to_num(self.firing_fields.smooth_rate_maps)
        for bin_num, (start_time, decoded_time, end_time) in enumerate(zip(start_times, self.decoded_times, end_times)):
            if bin_num % 1000 == 0:
                print(f'decoding at time {start_time:.2f}s (/{start_times[-1]:.2f}s)')
            position, run_type = self.tracking.at_time(decoded_time, d_runs=True)
            self.real_positions[bin_num] = position

            if run_type != -1:
                self.spike_counts[:, bin_num] = self.count_spikes(start_time, end_time)
                if np.sum(self.spike_counts[:, bin_num]) >= min_spikes:
                    self.decoded_probabilities[:, bin_num] = \
                        (np.prod(np.power(firing_rates[:, run_type], self.spike_counts[:, bin_num].reshape(-1, 1)), axis=0)
                         * np.exp(-(end_time - start_time) * np.sum(firing_rates[:, run_type], axis=0)))

                self.decoded_probabilities[:, bin_num] /= np.sum(self.decoded_probabilities[:, bin_num])

    def plot_decoded_probabilities(self, step_size, plot_most_probable_positions=False, plot_theta_peaks=True,
                                   lfp=None):
        """After decoding on sliding time windows, plots the results of the decoder.

        Args:
            step_size (float): Stride/step for the sliding window (s).
            plot_most_probable_positions (bool): Plot the positions of maximum decoded probability.
            plot_theta_peaks (bool): Plot vertical lines indicating the peaks of theta.
            lfp (LFP): LFP instance.
        """
        fig, ax = plt.subplots()
        mat = ax.matshow(self.decoded_probabilities, aspect="auto",  origin="lower", cmap="hot",
                         extent=(self.decoded_times[0] - step_size/2, self.decoded_times[-1] + step_size/2,
                                 self.firing_fields.positions[0] - self.firing_fields.bin_size / 2,
                                 self.firing_fields.positions[-1] + self.firing_fields.bin_size / 2))
        plt.colorbar(mat)
        ax.xaxis.set_ticks_position("bottom")

        if plot_theta_peaks and lfp is not None:
            for peak_time in lfp.times[lfp.cycle_boundaries[0]]:
                if self.decoded_times[0] < peak_time < self.decoded_times[-1]:
                    ax.axvline(peak_time)

        ax.plot(self.decoded_times, self.real_positions + self.tracking.d_runs_offset, 'C1', label="real position")

        if plot_most_probable_positions:
            decoded_positions = np.where(np.isnan(self.decoded_probabilities[0]), np.nan,
                                         np.argmax(self.decoded_probabilities, axis=0))
            ax.plot(self.decoded_times,
                    self.firing_fields.positions[0] + decoded_positions*self.firing_fields.bin_size, 'g.')

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position (cm)")
        ax.legend()
        self.maybe_save_fig(fig, "decoded_time_bins")

    def plot_spike_counts(self, window_size):
        """After decoding on sliding time windows, plots a heatmap of spike counts per decoding bin and a histogram
        showing the number of cells with over x spikes.

        Args:
            window_size (float): Size of the time window (s).
        """
        # heatmap of spike counts per decoding bin.
        fig, ax = plt.subplots()
        mat = ax.matshow(self.spike_counts, aspect="auto", origin="lower",
                         extent=(self.decoded_times[0] - window_size/2, self.decoded_times[-1] + window_size/2, -0.5,
                                 self.spike_counts.shape[0] + 0.5))
        ax.xaxis.set_ticks_position("bottom")
        bar = plt.colorbar(mat)
        bar.ax.set_ylabel("Spike count")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cell number")

        # histogram of number of cells with over x spikes
        max_spike_count = np.nanmax(self.spike_counts.astype(int))
        fig, ax = plt.subplots()
        for min_spike_count in range(1, max_spike_count + 1):
            cell_counts = np.zeros(self.decoded_times.size)
            for time_bin_num in range(self.decoded_times.size):
                cell_counts[time_bin_num] = np.sum(self.spike_counts[:, time_bin_num] >= min_spike_count)
            ax.plot(self.decoded_times, cell_counts, label=f">= {min_spike_count} spikes")
        ax.legend()
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Cell count")
        self.maybe_save_fig(fig, "spike_counts")

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):
        decoder = cls(super_group_name, group_name, child_name, kwargs['SpikesBase'].spikes, kwargs['Tracking'],
                      kwargs['FiringFields'], save_figures=save_figures, figure_format=figure_format,
                      figures_path=figures_path, pickle_results=pickle_results, pickles_path=pickles_path)
        decoder.decode_phase_bins(lfp=kwargs['LFP'], phase_bin_size=parameters_dict['phase_bin_size'],
                                  phase_step_size=parameters_dict['phase_step_size'],
                                  min_spikes=parameters_dict['min_spikes'], plot_decoded_probabilities=False)
        return decoder


