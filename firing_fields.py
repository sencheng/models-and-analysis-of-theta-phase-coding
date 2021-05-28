import os
import math
import pickle
import copy
import numpy as np
from scipy import signal
from scipy.stats import linregress
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from data_analysis.general import Base, nan_smooth
from data_analysis.spikes_basics import Spikes, SpikesBase
from data_analysis.tracking import Tracking
from data_analysis.lfp import LFP


class FiringFields(Base):
    """Calculate firing rate maps.

    Args:
        super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is defined
            as belonging to the super-group, it will be shared across sub-groups.
        group_name (string): Name of the low-level sub-group used for pickles and figures.
        child_name (string): Name of the instance used for pickles and figures.
        spikes (Spikes): Spikes instance.
        tracking (Tracking): Tracking instance.
        sigma (float): Standard deviation for the Gaussian filter (cm).
        save_figures (bool): Whether to save the figures.
    """
    dependencies = (LFP, SpikesBase, Tracking)

    def __init__(self, super_group_name, group_name, child_name, spikes, tracking, sigma, consecutive_nans_max,
                 fields_folder='fields', save_figures=False, figure_format="png", figures_path="figures",
                 pickle_results=True, pickles_path="pickles"):

        super().__init__(super_group_name, group_name, child_name, save_figures, figure_format, figures_path,
                         pickle_results, pickles_path)

        self.spikes = spikes
        self.tracking = tracking
        self.bin_size = tracking.spatial_bin_size
        self.num_bins = tracking.num_spatial_bins
        self.sigma = sigma
        self.consecutive_nans_max = consecutive_nans_max
        self.fields_path = f"{fields_folder}/{self.super_group_name}.{self.group_name}.pkl"
        self.positions = tracking.d_runs_offset + np.arange(self.num_bins) * self.bin_size
        self.electrode_cluster_pairs = spikes.electrode_cluster_pairs

        center_start = tracking.corner_size
        center_end = tracking.d_runs_span - tracking.corner_size

        # calculate time spent per bin
        occupancy = np.zeros((2, self.num_bins))
        sampling_time = 1/tracking.sampling_rate
        for d, run_type, speed, significant_theta in zip(tracking.d_runs, tracking.run_type, tracking.speed_1D,
                                                         tracking.significant_theta):
            if run_type != -1 and significant_theta:
                occupancy[run_type, int(round(d/self.bin_size))] += sampling_time

        # calculate spike counts per bin
        self.spike_counts = np.zeros((len(self.electrode_cluster_pairs), 2, self.num_bins))
        for pair_num, electrode_cluster_pair in enumerate(self.electrode_cluster_pairs):
            for spike_time in spikes.spike_times[pair_num]:
                d, run_type, speed = tracking.at_time(spike_time, d_runs=True, return_speed=True)
                if run_type != -1:
                    if (((center_start < d < center_end) and speed > tracking.min_central_speed)
                            or d < center_start or d > center_end):
                        self.spike_counts[pair_num, run_type, int(round(d/self.bin_size))] += 1

        # calculate firing rates
        self.rate_maps = np.full((len(self.electrode_cluster_pairs), 2, self.num_bins), np.nan)
        self.rate_maps[:, occupancy > 0] = self.spike_counts[:, occupancy > 0] / occupancy[occupancy > 0]

        self.smooth_rate_maps = np.empty(self.rate_maps.shape)
        for pair_num in range(len(self.electrode_cluster_pairs)):
            for run_type in range(2):
                self.smooth_rate_maps[pair_num, run_type] = nan_smooth(self.rate_maps[pair_num, run_type],
                                                                       sigma/self.bin_size)

        # for field candidates
        self.cand_pair_nums = []
        self.cand_run_types = []
        self.cand_bounds = []
        self.cand_bound_indices = []
        self.cand_bounds_ok = []
        self.cand_peak_indices = []
        self.cand_peak_rates = []
        self.cand_spikes_mean_speeds = []
        self.cand_characteristic_speeds = []
        self.cand_distances_from_start = []
        self.cand_distances_to_border = []

    @staticmethod
    def count_consecutive_false(array):
        first_true_index = np.argmax(array)
        if first_true_index:
            return first_true_index - 1
        else:
            return len(array) - 1

    def too_many_nans(self, not_nans, index):
        """Returns true if there are too many consecutive nans around some index.
        """
        if not not_nans[index]:
            consecutive_nans = 1
            consecutive_nans += self.count_consecutive_false(not_nans[index:])
            consecutive_nans += self.count_consecutive_false(not_nans[:index + 1][::-1])
            return consecutive_nans > self.consecutive_nans_max
        else:
            return False

    def end_of_field(self, smooth_rate_map, not_nans, peak_index, threshold_rate, increment, within_range,
                     enforce_unique_peak=False):
        index = peak_index + increment
        while True:
            if not within_range(index) or self.too_many_nans(not_nans, index):
                return index - increment, 0
            elif enforce_unique_peak and smooth_rate_map[index] > smooth_rate_map[peak_index]:
                return -1, -1
            elif smooth_rate_map[index] < threshold_rate:
                return index, 1
            index += increment

    def find_field_bounds(self, group_smooth_rates, peak_index, peak_rate, threshold, not_nans,
                          enforce_unique_peak=False):
        threshold_rate = peak_rate * threshold
        start, start_ok = self.end_of_field(group_smooth_rates, not_nans, peak_index, threshold_rate, increment=-1,
                                            within_range=lambda x: x >= 0, enforce_unique_peak=enforce_unique_peak)
        end, end_ok = self.end_of_field(group_smooth_rates, not_nans, peak_index, threshold_rate, increment=1,
                                        within_range=lambda x: x < self.num_bins,
                                        enforce_unique_peak=enforce_unique_peak)
        return [start, end], [start_ok, end_ok]

    def peak_distance_from_start(self, field_peak_index, run_type):
        if run_type == 0:
            return self.positions[field_peak_index] - self.tracking.d_runs_offset
        else:
            return self.tracking.d_runs_offset + self.tracking.d_runs_span - self.positions[field_peak_index]

    def peak_distance_to_border(self, field_peak_index):
        peak_position = self.positions[field_peak_index]
        return min(peak_position, self.tracking.d_runs_offset + self.tracking.d_runs_span - peak_position)

    def find_fields_candidates(self, min_spikes, min_firing_rate, threshold, peak_prominence_threshold):
        """Find the extent of place field candidates based on a minimum peak firing rate and a threshold.

        Args:
            min_spikes (int): Minimum number of spikes fields need to have.
            min_firing_rate (float): Minimum peak firing rate to classify as place field (Hz).
            threshold (float): Proportion of the peak firing rate down to which the place field is defined.
            peak_prominence_threshold (float): Proportion of the peak firing rate that sets the minimum
                prominence of the peak.
        """
        for pair_num in range(len(self.electrode_cluster_pairs)):
            for run_type in range(2):
                peak_indices = signal.find_peaks(self.smooth_rate_maps[pair_num, run_type], height=min_firing_rate)[0]
                not_nans = ~np.isnan(self.rate_maps[pair_num][run_type])
                for peak_index in peak_indices:
                    peak_rate = self.smooth_rate_maps[pair_num, run_type, peak_index]
                    bound_indices, bounds_ok = self.find_field_bounds(self.smooth_rate_maps[pair_num, run_type],
                                                                      peak_index, peak_rate, threshold, not_nans,
                                                                      enforce_unique_peak=True)

                    if -1 not in bounds_ok and sum(bounds_ok) > 0:
                        prominence_threshold = peak_rate * (1 - peak_prominence_threshold)
                        prominence_ok = ((self.smooth_rate_maps[pair_num, run_type, bound_indices[0]:peak_index+1]
                                          <= prominence_threshold).any() and
                                         (self.smooth_rate_maps[pair_num, run_type, peak_index:bound_indices[1]+1]
                                          < prominence_threshold).any())
                        if not prominence_ok:
                            continue

                        if np.sum(self.spike_counts[pair_num, run_type,
                                  bound_indices[0]:bound_indices[1]+1]) < min_spikes:
                            continue

                        # start_position = self.positions[bound_indices[0]] if bounds_ok[0] > 0 else np.nan
                        # end_position = self.positions[bound_indices[1]] if bounds_ok[1] > 0 else np.nan

                        self.cand_pair_nums.append(pair_num)
                        self.cand_run_types.append(run_type)
                        self.cand_bounds_ok.append(bounds_ok)
                        self.cand_bound_indices.append(bound_indices)
                        self.cand_bounds.append((self.positions[bound_indices[0]], self.positions[bound_indices[1]]))
                        self.cand_peak_indices.append(peak_index)
                        self.cand_peak_rates.append(peak_rate)

                        field_speeds = []
                        lower_bound = self.positions[bound_indices[0]]
                        upper_bound = self.positions[bound_indices[1]]
                        for spike_time in self.spikes.spike_times[pair_num]:
                            d, spike_run_type, speed = self.tracking.at_time(spike_time, d_runs=True, return_speed=True)
                            if spike_run_type == run_type and (lower_bound < d < upper_bound):
                                field_speeds.append(speed)
                        self.cand_spikes_mean_speeds.append(np.mean(field_speeds))
                        self.cand_characteristic_speeds.append(
                            np.nanmean(self.tracking.characteristic_speeds[run_type][bound_indices[0]:bound_indices[1]+1]))

                        self.cand_distances_from_start.append(self.peak_distance_from_start(peak_index, run_type))
                        self.cand_distances_to_border.append(self.peak_distance_to_border(peak_index))

    def save_ok_fields(self, field_nums, fields_folder_path="fields"):

        fields = {'pair_nums': [], 'run_types': [], 'bounds_ok': [], 'bound_indices': [], 'bounds': [],
                  'peak_indices': [], 'peak_rates': [], 'spikes_mean_speeds': [], 'characteristic_speeds': [],
                  'distances_from_start': [], 'distances_to_border': [], 'complete': []}

        for field_num in field_nums:
            if np.sum(self.cand_bounds_ok[field_num]) == 2:
                complete = True
            else:
                complete = False

            fields['pair_nums'].append(self.cand_pair_nums[field_num])
            fields['run_types'].append(self.cand_run_types[field_num])
            fields['bounds_ok'].append(self.cand_bounds_ok[field_num])
            fields['bound_indices'].append(self.cand_bound_indices[field_num])
            fields['bounds'].append(self.cand_bounds[field_num])
            fields['peak_indices'].append(self.cand_peak_indices[field_num])
            fields['peak_rates'].append(self.cand_peak_rates[field_num])
            fields['spikes_mean_speeds'].append(self.cand_spikes_mean_speeds[field_num])
            fields['characteristic_speeds'].append(self.cand_characteristic_speeds[field_num])
            fields['distances_from_start'].append(self.cand_distances_from_start[field_num])
            fields['distances_to_border'].append(self.cand_distances_to_border[field_num])
            fields['complete'].append(complete)

        if not os.path.isdir(fields_folder_path):
            os.mkdir(fields_folder_path)

        with open(self.fields_path, 'wb') as fields_file:
            pickle.dump(fields, fields_file)

    def accept_all(self):
        self.save_ok_fields(range(len(self.cand_pair_nums)))

    def screen_fields(self, lfp, rewrite=False):
        """For all putative place fields, show firing rate and phase vs. position plots for manually accepting or
        discarding them.

        Args:
            lfp (LFP):
            rewrite (bool):
        """
        if not os.path.exists(self.fields_path) or rewrite:
            field_num = 0
            pair_num = 0
            run_type = 0
            field_bounds = ()
            bounds_ok = ()

            field_nums = []

            fig, ax = plt.subplots(2, sharex="col")
            plt.subplots_adjust(bottom=0.25)

            def get_fields():
                nonlocal field_num, pair_num, run_type, field_bounds, bounds_ok
                for field_num, (pair_num, run_type, field_bounds, bounds_ok) in \
                        enumerate(zip(self.cand_pair_nums, self.cand_run_types, self.cand_bounds, self.cand_bounds_ok)):
                    yield

            def plot_next():
                nonlocal field_num, pair_num, run_type
                try:
                    next(candidate_generator)
                    ax[0].clear()
                    ax[0].set_title(f"{field_num}, {'→' if run_type == 0 else '←'}")
                    ax[0].plot(self.positions, self.smooth_rate_maps[pair_num, run_type])
                    if bounds_ok[0]:
                        ax[0].axvline(field_bounds[0], color='C1', linestyle="dotted")
                    if bounds_ok[1]:
                        ax[0].axvline(field_bounds[1], color='C1', linestyle="dotted")
                    ax[0].set_ylabel("Firing rate (Hz)")

                    in_field_positions = []
                    in_field_phases = []
                    in_field_speeds = []
                    out_of_field_positions = []
                    out_of_field_phases = []
                    electrode_index = self.spikes.electrodes.index(self.electrode_cluster_pairs[pair_num][0])

                    for spike_time in self.spikes.spike_times[pair_num]:
                        position, spike_run_type, speed = self.tracking.at_time(spike_time, return_speed=True)
                        if spike_run_type == run_type:
                            phase, = lfp.at_time(spike_time, electrode_index, return_phase=True)

                            out_of_field = ((~np.isnan(field_bounds[0]) and position < field_bounds[0])
                                            or (~np.isnan(field_bounds[1]) and position > field_bounds[1]))

                            if out_of_field:
                                out_of_field_positions.append(position)
                                out_of_field_phases.append(phase)
                            else:
                                in_field_positions.append(position)
                                in_field_phases.append(phase)
                                in_field_speeds.append(speed)

                    ax[1].clear()
                    ax[1].scatter(in_field_positions, in_field_phases, c=in_field_speeds,
                                  s=plt.rcParams['lines.markersize'], vmin=0, vmax=np.nanmax(self.tracking.speed_1D))
                    ax[1].plot(out_of_field_positions, out_of_field_phases, '.', color='C7')
                    ax[1].set_xlabel("Position (cm)")
                    ax[1].set_ylabel("Phase (deg)")
                    fig.align_ylabels(ax)
                    return False

                except StopIteration:
                    return True

            def ok(_):
                field_nums.append(field_num)
                go_on()

            def go_on(event=None):
                finished = plot_next()
                if finished:
                    self.save_ok_fields(field_nums)
                    plt.close(fig)

            candidate_generator = get_fields()
            plot_next()

            ax_ok = plt.axes([0.65, 0.05, 0.1, 0.075])
            ax_discard = plt.axes([0.80, 0.05, 0.1, 0.075])
            button_ok = Button(ax_ok, 'OK')
            button_ok.on_clicked(ok)
            button_discard = Button(ax_discard, 'Discard')
            button_discard.on_clicked(go_on)
            plt.show()

    def plot_traces(self, field_nums):
        """Plot firing rate maps as traces.

        Args:
            field_nums: Field numbers to plot.
        """
        run_types = ['Forward run', 'Backward run']
        fig, ax = plt.subplots(2, 1, sharex='col')
        for field_num in field_nums:
            pair_num = self.cand_pair_nums[field_num]
            run_type = self.cand_run_types[field_num]
            ax[run_type].plot(self.positions, self.rate_maps[pair_num, run_type], label=field_num)
            ax[run_type].plot(self.positions, self.smooth_rate_maps[pair_num, run_type],
                              label=f"{field_num}, smooth")
            ax[run_type].set_title(run_types[run_type])
            ax[run_type].legend(loc='lower left')
            ax[run_type].set_ylabel('Firing rate (Hz)')
        ax[1].set_xlabel('Displacement (cm)')
        self.maybe_save_fig(fig, "firing_rate")

    def plot_heatmap(self, smoothed=True, fig_size=(9, 4.5), verbose=False):
        """Plot all firing rate maps as a heatmap, with cells sorted by the location of their peak firing rate.

        Args:
            smoothed (bool): Plot smoothed or raw firing rates.
            fig_size (tuple(float)): Figure size.
            verbose (bool): Print sorted cell ids.
        """
        run_types = ['Forward run', 'Backward run']
        rate_maps = self.smooth_rate_maps if smoothed else self.rate_maps

        fig, ax = plt.subplots(1, 2, constrained_layout=True, figsize=fig_size)
        extent = (self.positions[0] - self.bin_size / 2, self.positions[-1] + self.bin_size / 2,
                  len(self.electrode_cluster_pairs) - 0.5, -0.5)
        for run_type, run_type_name in enumerate(run_types):
            # sort cells by location of peak firing
            indices = range(len(self.electrode_cluster_pairs))
            sorted_indices = sorted(indices, key=lambda i: np.nanargmax(rate_maps[i, run_type]))
            sorted_pairs = np.array(self.electrode_cluster_pairs)[sorted_indices]
            if verbose:
                print(f'\nSorted (electrode, cluster_id) pairs in {run_type_name.lower()}:')
                for cell_num, sorted_pair in enumerate(sorted_pairs):
                    print(f'cell: {cell_num}, electrode: {sorted_pair[0]}, cluster_id: {sorted_pair[1]}')

            mat = ax[run_type].matshow(rate_maps[sorted_indices, run_type, :], extent=extent, aspect="auto", vmin=0,
                                       vmax=np.nanmax(rate_maps))
            ax[run_type].xaxis.set_ticks_position('bottom')
            ax[run_type].set_title(run_type_name)
            ax[run_type].set_xlabel('Displacement (cm)')
            ax[run_type].set_ylabel('Sorted cell number')

            if run_type == 1:
                bar = fig.colorbar(mat, ax=ax[run_type], aspect=40)
                bar.ax.set_ylabel("Firing rate (Hz)")

        self.maybe_save_fig(fig, f"heatmaps")

    def field_size(self, bound_indices, bounds_ok, peak_index):
        if sum(bounds_ok) == 2:
            return self.positions[bound_indices[1]] - self.positions[bound_indices[0]]
        else:
            field_bound_index = bound_indices[0] if bounds_ok[0] else bound_indices[1]
            return 2 * abs(field_bound_index - peak_index) * self.bin_size

    def occupancy_by_speed(self, speed_groups):
        # calculate time spent per bin
        occupancy = np.zeros((2, len(speed_groups), self.num_bins))
        sampling_time = 1 / self.tracking.sampling_rate
        for d, run_type, speed, significant_theta in zip(self.tracking.d_runs, self.tracking.run_type,
                                                         self.tracking.speed_1D, self.tracking.significant_theta):
            if run_type != -1 and significant_theta:
                if run_type == 1:
                    speed *= -1
                for speed_group_num, speed_group in enumerate(speed_groups):
                    if speed_group[0] < speed < speed_group[1]:
                        occupancy[run_type, speed_group_num, int(round(d / self.bin_size))] += sampling_time
        smooth_occupancy = gaussian_filter1d(occupancy, self.sigma / self.bin_size, mode='nearest')
        return occupancy, smooth_occupancy, np.max(smooth_occupancy, axis=(1, 2))

    @staticmethod
    def max_summed_distances(pattern_size):
        """Sum of pairwise distances between pattern elements."""
        sum_distance = 0
        for n in range(2, pattern_size + 1):
            sum_distance += (n - 1) * n / 2
        return sum_distance + 1

    @staticmethod
    def spread(pattern, max_sum_distance):
        total_spread = 0
        for i in range(len(pattern)):
            if pattern[i]:
                for j in range(i + 1, len(pattern)):
                    if pattern[j]:
                        total_spread += j - i
        return total_spread / max_sum_distance

    def fields_spread(self, run_types, bound_indices, occupancy, min_occupancy):
        spreads = []
        for run_type, field_bound_indices in zip(run_types, bound_indices):
            field_bound_indices[1] += 1

            max_summed_distances = self.max_summed_distances(field_bound_indices[1] - field_bound_indices[0])

            field_spreads = []
            for speed_group_occupancy in occupancy[run_type]:
                ok_field_occupancy = speed_group_occupancy[slice(*field_bound_indices)] > min_occupancy
                field_spread = self.spread(ok_field_occupancy, max_summed_distances)
                field_spreads.append(field_spread)
            spreads.append(field_spreads)
        return np.array(spreads)

    def screened_fields(self, include_incomplete=False):
        with open(self.fields_path, 'rb') as fields_file:
            all_fields = pickle.load(fields_file)

        fields = {}
        for key in all_fields:
            fields[key] = [all_fields[key][i] for i, complete in enumerate(all_fields['complete'])
                           if complete or include_incomplete]
        return fields

    def ok_speed_bins_histogram(self, speed_groups, min_occupancy, min_spread, hist_bin_size, num_hist_bins,
                                verbose=False):
        central_speeds = np.mean(speed_groups, axis=1)

        # calculate for which speed bins the spread of occupancy is enough for each field
        fields = self.screened_fields()
        occupancy, smooth_occupancy, max_smooth_occupancies = self.occupancy_by_speed(speed_groups)
        fields_spread_ok = self.fields_spread(fields['run_types'], fields['bound_indices'], occupancy,
                                              min_occupancy) > min_spread

        # calculate distance of ok speed bins to characteristic speed
        overall_field_num = 0
        ok_by_speed_distance = [[] for _ in range(num_hist_bins)]
        for c_speed in fields['characteristic_speeds']:
            if speed_groups[0][0] < c_speed < speed_groups[-1][-1]:
                for speed_bin_num, field_spread_ok in enumerate(fields_spread_ok[overall_field_num]):
                    speed_distance = abs(c_speed - central_speeds[speed_bin_num])
                    hist_bin_num = int(speed_distance / hist_bin_size)
                    if hist_bin_num < num_hist_bins:
                        ok_by_speed_distance[hist_bin_num].append(field_spread_ok)
            overall_field_num += 1

        if verbose:
            for distance_num, oks in enumerate(ok_by_speed_distance):
                print(f"{np.mean(oks):.2f} ok with {len(oks)} points for a distance "
                      f"of {(distance_num + 0.5) * hist_bin_size}")

        self.maybe_pickle_results(ok_by_speed_distance, "ok_by_speed_distance")

    def field_sizes_by_speed(self, speed_groups, min_peak_firing_rate, threshold, peak_prominence_threshold,
                             min_occupancy=0, min_spread=0, plot_fields=False, fields_per_plot=6, fig_size=(10, 6),
                             constrained_layout=False, field_nums=None):
        """Calculate place field sizes pooling spikes from different running speeds.

        Args:
            speed_groups (list(list(float))): List of (lower, upper) bounds for speed groups.
            min_peak_firing_rate (float): Minimum peak firing rate to classify as place field (Hz).
            threshold (float): Proportion of the peak firing rate down to which the place field is defined.
            peak_prominence_threshold (float): Proportion of the peak that defines the minimum peak prominence required
                for considering a field for half-size calculation.
            min_occupancy (float): Minimum time spent per bin (s).
            min_spread (float): Minimum value for the spread of valid occupancy bins [0-1].
            plot_fields (bool): Whether to plot the fields.
            fields_per_plot (int): Number of place fields per figure.
            fig_size (tuple(float)): Size of the figure in inches.
            constrained_layout (bool): For matplotlib. Very slow but otherwise figures are screwed.
            field_nums (list(int)): Fields to analyze.
        """
        fields = self.screened_fields(include_incomplete=True)
        if field_nums is not None:
            for key, values in fields.items():
                fields[key] = [value for i, value in enumerate(values) if i in field_nums]

        occupancy, smooth_occupancy, max_smooth_occupancies = self.occupancy_by_speed(speed_groups)
        fields_spread = self.fields_spread(fields['run_types'], fields['bound_indices'], occupancy, min_occupancy)

        # calculate rate maps
        rate_maps = np.full((len(fields['pair_nums']), len(speed_groups), self.num_bins), np.nan)
        ok_rate_maps = np.full((len(fields['pair_nums']), len(speed_groups), self.num_bins), np.nan)
        previous_pair_num = None
        previous_run_type = None
        for field_num, (pair_num, target_run_type) in enumerate(zip(fields['pair_nums'], fields['run_types'])):
            if pair_num == previous_pair_num and target_run_type == previous_run_type:
                rate_maps[field_num] = rate_maps[field_num - 1]
                ok_rate_maps[field_num] = ok_rate_maps[field_num - 1]
            else:
                spike_counts = np.zeros((len(speed_groups), self.num_bins))
                for spike_time in self.spikes.spike_times[pair_num]:
                    d, run_type, speed = self.tracking.at_time(spike_time, d_runs=True, return_speed=True)
                    if run_type == target_run_type:
                        for speed_group_num, speed_group in enumerate(speed_groups):
                            if speed_group[0] < speed < speed_group[1]:
                                spike_counts[speed_group_num, int(round(d / self.bin_size))] += 1

                occupancy_positive = occupancy[target_run_type] > 0

                rate_maps[field_num, occupancy_positive] = (spike_counts[occupancy_positive]
                                                            / occupancy[target_run_type][occupancy_positive])
                occupancy_ok = occupancy[target_run_type] > min_occupancy
                ok_rate_maps[field_num, occupancy_ok] = (spike_counts[occupancy_ok]
                                                         / occupancy[target_run_type][occupancy_ok])
            previous_pair_num = pair_num
            previous_run_type = target_run_type

        # prepare figures
        if plot_fields:
            axes = []
            figs = []
            for plot_num in range(math.ceil(len(fields['pair_nums']) / fields_per_plot)):
                fig, ax = plt.subplots(len(speed_groups)+1, fields_per_plot, figsize=fig_size,
                                       constrained_layout=constrained_layout)
                # ax[-2, 0].set_xlabel("Position (cm)")
                # ax[-2, 0].set_ylabel("Firing\nrate (Hz)")
                # ax[-1, 0].set_xlabel("Running\nspeed (cm/s)")
                # ax[-1, 0].set_ylabel("Place field\nsize (cm)")
                axes.append(ax)
                figs.append(fig)

        # calculate place field sizes
        all_field_sizes = np.full((len(fields['pair_nums']), len(speed_groups)), np.nan)
        all_field_peak_rates = np.full((len(fields['pair_nums']), len(speed_groups)), np.nan)
        average_speeds = np.mean(speed_groups, axis=1)

        for field_num, (run_type, bound_indices) in \
                enumerate(zip(fields['run_types'], fields['bound_indices'])):
            lower_bound_index, upper_bound_index = bound_indices
            within_field = slice(lower_bound_index, upper_bound_index+1)

            for speed_group_num, speed_group in enumerate(speed_groups):
                if plot_fields:
                    ax = axes[int(field_num / fields_per_plot)][speed_group_num, field_num % fields_per_plot]
                    ax.spines['top'].set_visible(False)
                    axr = ax.twinx()
                    axr.spines['top'].set_visible(False)
                    axr.plot(self.positions, smooth_occupancy[run_type, speed_group_num], color='C7', linewidth=0.6)
                    axr.set_ylim(top=max(max_smooth_occupancies) * 1.1)
                    ax.spines['right'].set_edgecolor('C7')
                    axr.tick_params(axis='y', colors='C7')
                    if not field_num % fields_per_plot == fields_per_plot - 1:
                        axr.set_yticklabels([])
                    ax.set_zorder(1)
                    ax.patch.set_visible(False)
                    ax.plot(self.positions, rate_maps[field_num, speed_group_num], '.', color='C0', markersize=3)
                    max_rate = 1.1*np.nanmax(rate_maps[field_num])
                    ax.set_ylim([-0.05*max_rate, max_rate])
                    # # for distinguishing valid and invalid points, plot above in red and uncomment here
                    # ax.plot(self.positions, ok_rate_maps[field_num, speed_group_num], '.', color='C0', markersize=4)
                    for i in range(2):
                        if fields['bounds_ok'][field_num][i]:
                            ax.axvline(fields['bounds'][field_num][i], color='C7', linestyle='dashed')
                    # ax.annotate(f"{fields_spread[field_num, speed_group_num]:.2f}", (0.75, 0.8),
                    #             xycoords="axes fraction", fontsize="x-small")

                    align_yaxis(ax, 0, axr, 0)

                    if speed_group_num == 0:
                        ax.set_title(f"field {field_num}", fontsize='medium')
                    if field_num % fields_per_plot == 0:
                        ax.annotate(f"{speed_group}\ncm/s", (-0.85, 0.5), xycoords="axes fraction",
                                    rotation="vertical", va='center', multialignment="center")

                    if speed_group_num < len(speed_groups) - 1:
                        ax.set_xticklabels([])
                    # elif field_num % fields_per_plot == fields_per_plot - 1:
                    #     axr.set_ylabel("Occupancy\n(s)", color='C7')

                if np.sum(~np.isnan(rate_maps[field_num, speed_group_num][within_field])) > 2:
                    group_smooth_rates = nan_smooth(rate_maps[field_num, speed_group_num], self.sigma / self.bin_size)

                    if plot_fields:
                        ax.plot(self.positions, group_smooth_rates)

                    if fields_spread[field_num, speed_group_num] > min_spread:
                        field_rates = group_smooth_rates[within_field]
                        peak_indices, peak_properties = signal.find_peaks(field_rates,
                                                                          # height=global_threshold_rate,
                                                                          height=min_peak_firing_rate,
                                                                          prominence=(np.nanmax(field_rates)
                                                                                      * peak_prominence_threshold))
                        if len(peak_indices):
                            peak_index = peak_indices[np.argmax(peak_properties['peak_heights'])] + within_field.start

                            if (np.isnan(group_smooth_rates[peak_index:]).all()
                                    or np.isnan(group_smooth_rates[:peak_index]).all()):
                                continue

                            peak_rate = group_smooth_rates[peak_index]
                            all_field_peak_rates[field_num, speed_group_num] = peak_rate

                            not_nans = ~np.isnan(ok_rate_maps[field_num, speed_group_num])
                            bounds, bounds_ok = self.find_field_bounds(group_smooth_rates, peak_index, peak_rate,
                                                                       threshold, not_nans, enforce_unique_peak=True)
                            if -1 not in bounds_ok and sum(bounds_ok) > 0:
                                all_field_sizes[field_num, speed_group_num] = self.field_size(bounds, bounds_ok,
                                                                                              peak_index)
                                if plot_fields:
                                    ax.plot(self.positions[peak_index], peak_rate, '*')
                                    for bound_num in range(2):
                                        if bounds_ok[bound_num]:
                                            ax.axvline(self.positions[bounds[bound_num]], color='k')

            if plot_fields:
                ax = axes[int(field_num / fields_per_plot)][-1, field_num % fields_per_plot]
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                field_sizes = all_field_sizes[field_num]
                ax.plot(average_speeds, field_sizes, '.', color='k')
                sizes_ok = ~np.isnan(field_sizes)
                if np.sum(sizes_ok) > 1:
                    fit = linregress(average_speeds[sizes_ok], field_sizes[sizes_ok])
                    ax.plot(average_speeds[sizes_ok], average_speeds[sizes_ok] * fit.slope + fit.intercept, 'C7')
                    size_range = np.nanmax(field_sizes) - np.nanmin(field_sizes)
                    if size_range > 0:
                        ax.set_ylim([np.nanmin(field_sizes) - 0.05*size_range,
                                     np.nanmax(field_sizes) + 0.05*size_range])

        self.maybe_pickle_results(all_field_sizes, "field_sizes", subfolder="per_speed")
        self.maybe_pickle_results(all_field_peak_rates, "field_peak_rates", subfolder="per_speed")

        if plot_fields:
            for fig_num, fig in enumerate(figs):
                fig.tight_layout(h_pad=0.1, w_pad=0.4)
                self.maybe_save_fig(fig, f"batch_{fig_num}")

        fig, ax = plt.subplots()
        for field_sizes in all_field_sizes:
            ax.plot(average_speeds, field_sizes, '*-')
        ax.set_ylabel("Field size (cm)")
        ax.set_xlabel("Running speed (cm/s)")
        plt.tight_layout()
        self.maybe_save_fig(fig, "field_sizes")

    def field_sizes_vs_stuff(self, max_marker_size=30, fig_size=(12, 3), plot=False):

        sizes = []
        spikes_mean_speeds = []
        characteristic_speeds = []
        peak_distances_from_start = []
        peak_normalized_pos = []
        peak_distances_to_border = []
        peak_rates = []

        fields = self.screened_fields(include_incomplete=True)
        for field_num in range(len(fields['pair_nums'])):
            field_bounds_indices = fields['bound_indices'][field_num]
            field_peak_index = fields['peak_indices'][field_num]
            sizes.append(self.field_size(field_bounds_indices, fields['bounds_ok'][field_num], field_peak_index))
            spikes_mean_speeds.append(fields['spikes_mean_speeds'][field_num])
            characteristic_speeds.append(fields['characteristic_speeds'][field_num])
            peak_distances_from_start.append(fields['distances_from_start'][field_num])
            peak_normalized_pos.append(peak_distances_from_start[-1] / self.tracking.d_runs_span)
            peak_distances_to_border.append(fields['distances_to_border'][field_num])
            peak_rates.append(fields['peak_rates'][field_num])

        self.maybe_pickle_results([sizes], "sizes")
        self.maybe_pickle_results([spikes_mean_speeds], "spikes_mean_speeds")
        self.maybe_pickle_results([characteristic_speeds], "characteristic_speeds")
        self.maybe_pickle_results([peak_distances_from_start], "peak_distances_from_start")
        self.maybe_pickle_results([peak_normalized_pos], "peak_normalized_pos")
        self.maybe_pickle_results([peak_distances_to_border], "peak_distances_to_border")
        self.maybe_pickle_results([peak_rates], "peak_rates")

        if plot:
            fig, ax = plt.subplots(1, 4, sharey='row', figsize=fig_size)
            weights = np.array(peak_rates)/max(peak_rates)*max_marker_size
            ax[0].scatter(peak_distances_from_start, sizes, s=weights)
            ax[0].set_xlabel("Distance from\nthe start of the run (cm)")
            ax[0].set_ylabel("Field size (cm)")
            ax[1].scatter(peak_distances_to_border, sizes, s=weights)
            ax[1].set_xlabel("Distance to\nthe nearest border (cm)")
            ax[2].scatter(spikes_mean_speeds, sizes, s=weights)
            ax[2].set_xlabel("Mean speed\nfor field's spikes (cm/s)")
            ax[3].scatter(characteristic_speeds, sizes, s=weights)
            ax[3].set_xlabel("Characteristic speed\nthrough the field (cm/s)")
            plt.tight_layout()
            self.maybe_save_fig(fig, "field_sizes_vs_stuff")

    @staticmethod
    def field_skewness(firing_rates):
        x = np.arange(1, len(firing_rates) + 1)
        center_of_mass = np.sum(x*firing_rates) / np.sum(firing_rates)
        m3 = np.sum(firing_rates * (x - center_of_mass)**3) / np.sum(firing_rates)
        std = np.sqrt(np.sum(firing_rates * (x - center_of_mass)**2) / np.sum(firing_rates))
        return m3 / std**3

    def field_skewness_vs_acceleration(self):
        fields = self.screened_fields(include_incomplete=False)
        accelerations = []
        skewnesses = []

        for pair_num, run_type, bound_indices in zip(fields['pair_nums'], fields['run_types'], fields['bound_indices']):
            lower_bound, upper_bound = bound_indices
            accelerations.append(np.mean(self.tracking.mean_acceleration[run_type][lower_bound:upper_bound + 1]))
            field_rates = self.smooth_rate_maps[pair_num][run_type][lower_bound:upper_bound + 1]
            skewnesses.append((1 - 2*run_type) * self.field_skewness(field_rates))

        fig, ax = plt.subplots()
        ax.plot(accelerations, skewnesses, '.')
        ax.axvline(0, linestyle='dashed', color='C7')
        ax.axhline(0, linestyle='dashed', color='C7')
        ax.set_xlabel(r"Mean acceleration $(cm/s^2)$")
        ax.set_ylabel("Skewness")
        self.maybe_save_fig(fig, "skewness_vs_acceleration")

        self.maybe_pickle_results([accelerations], "accelerations")
        self.maybe_pickle_results([skewnesses], "skewnesses")

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):

        firing_fields = cls(super_group_name, group_name, child_name, kwargs['SpikesBase'].spikes,
                            kwargs['Tracking'], parameters_dict['firing_rate_sigma'],
                            parameters_dict['consecutive_nans_max'], save_figures=save_figures,
                            figure_format=figure_format, figures_path=figures_path, pickle_results=pickle_results,
                            pickles_path=pickles_path)

        if not os.path.exists(firing_fields.fields_path):
            firing_fields.find_fields_candidates(parameters_dict['min_spikes'], parameters_dict['min_peak_firing_rate'],
                                                 parameters_dict['firing_rate_threshold'],
                                                 parameters_dict['peak_prominence_threshold'])
            if group_name == "EXPERIMENTAL":
                firing_fields.screen_fields(kwargs['LFP'])
            else:
                firing_fields.accept_all()

        return firing_fields


def align_yaxis(ax1, v1, ax2, v2):
    """adjust ax2 ylimit so that v2 in ax2 is aligned to v1 in ax1"""
    _, y1 = ax1.transData.transform((0, v1))
    _, y2 = ax2.transData.transform((0, v2))
    inv = ax2.transData.inverted()
    _, dy = inv.transform((0, 0)) - inv.transform((0, y1 - y2))
    miny, maxy = ax2.get_ylim()
    ax2.set_ylim(miny + dy, maxy + dy)