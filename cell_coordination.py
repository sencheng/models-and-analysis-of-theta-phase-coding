import numpy as np
import matplotlib.pyplot as plt
from data_analysis.general import Base
from data_analysis.lfp import LFP
from data_analysis.firing_fields import FiringFields


class CellCoordination(Base):

    dependencies = (LFP, FiringFields)

    def __init__(self, super_group_name, group_name, child_name, lfp, firing_fields, min_overlap, extra_margin,
                 save_figures=False, figure_format="png", figures_path="figures", pickle_results=True,
                 pickles_path="pickles"):

        super().__init__(super_group_name, group_name, child_name, save_figures, figure_format, figures_path,
                         pickle_results, pickles_path)

        self.lfp = lfp
        self.firing_fields = firing_fields

        self.fields = firing_fields.screened_fields()

        self.overlapping_field_nums = []
        self.overlaps = []
        self.find_field_overlaps(min_overlap, extra_margin)

        self.phase_differences = [[] for _ in range(len(self.overlapping_field_nums))]
        self.cycle_speeds = [[] for _ in range(len(self.overlapping_field_nums))]
        self.calculate_phase_differences()

    def find_field_overlaps(self, min_overlap, extra_margin, print_overlaps=False):
        """Find place field pairs which overlap and define the regions in which they overlap + margins on each side
        """
        for first_field_num, (run_type, first_field_bounds) in enumerate(zip(self.fields['run_types'],
                                                                             self.fields['bounds'])):
            for second_field_num in range(first_field_num + 1, len(self.fields['pair_nums'])):
                if self.fields['run_types'][second_field_num] == run_type:
                    second_field_bounds = self.fields['bounds'][second_field_num]
                    if first_field_bounds[0] < second_field_bounds[0]:
                        left_field_bounds = first_field_bounds
                        right_field_bounds = second_field_bounds
                        overlapping_field_nums = (first_field_num, second_field_num)
                    else:
                        left_field_bounds = second_field_bounds
                        right_field_bounds = first_field_bounds
                        overlapping_field_nums = (second_field_num, first_field_num)

                    if run_type == 1:
                        overlapping_field_nums = overlapping_field_nums[::-1]

                    if left_field_bounds[1] >= right_field_bounds[0] + min_overlap:
                        self.overlapping_field_nums.append(overlapping_field_nums)
                        left = max(0, right_field_bounds[0] - extra_margin)
                        right = min(self.firing_fields.positions[-1],
                                    min(left_field_bounds[1], right_field_bounds[1]) + extra_margin)
                        self.overlaps.append((left, right))

                        if print_overlaps:
                            print(f"field {self.overlapping_field_nums[-1][0]}: {left_field_bounds} and field "
                                  f"{self.overlapping_field_nums[-1][1]}: {right_field_bounds} "
                                  f"overlap between {self.overlaps[-1]}")

    def calculate_phase_differences(self, channel_index=0):
        cycle_boundaries = self.lfp.cycle_boundaries[channel_index][self.lfp.significant_cycles[channel_index]]
        left_boundaries = self.lfp.times[cycle_boundaries[:, 0]]
        right_boundaries = self.lfp.times[cycle_boundaries[:, 1]]

        for left_boundary, right_boundary in zip(left_boundaries, right_boundaries):
            central_position, run_type, cycle_speed = \
                self.firing_fields.tracking.at_time((left_boundary + right_boundary) / 2, return_speed=True)

            if run_type != -1:
                spike_phases = {}
                for overlap_num, (overlapping_field_nums, overlap) in \
                        enumerate(zip(self.overlapping_field_nums, self.overlaps)):
                    if overlap[0] <= central_position <= overlap[1]:
                        # collect spike phases
                        for overlapping_field_num in overlapping_field_nums:
                            if overlapping_field_num not in spike_phases:
                                pair_num = self.fields['pair_nums'][overlapping_field_num]
                                spike_times = self.firing_fields.spikes.spike_times[pair_num]
                                start, stop = np.searchsorted(spike_times, [left_boundary, right_boundary])
                                field_spike_phases = []
                                for spike_time in spike_times[start:stop]:
                                    field_spike_phases.append(
                                        self.lfp.at_time(spike_time, channel_index, return_phase=True)[0])
                                spike_phases[overlapping_field_num] = np.array(field_spike_phases)

                        # calculate phase differences
                        if (spike_phases[overlapping_field_nums[0]].size != 0
                                and spike_phases[overlapping_field_nums[1]].size != 0):

                            self.cycle_speeds[overlap_num].append(cycle_speed)
                            self.phase_differences[overlap_num].append((spike_phases[overlapping_field_nums[1]] -
                                                                        spike_phases[overlapping_field_nums[0]]
                                                                        [np.newaxis].T).flatten() % 360)

    @staticmethod
    def measure_variance(phase_differences):
        phase_differences = np.array(phase_differences) / 180 * np.pi
        r = np.sqrt(np.sum(np.cos(phase_differences))**2 + np.sum(np.sin(phase_differences))**2)/phase_differences.size
        return 1 - r

    @staticmethod
    def subsampled_variance(phase_differences, sample_size=20):
        if len(phase_differences) >= sample_size:
            phase_differences = np.array(phase_differences) / 180 * np.pi
            phase_differences = np.random.choice(phase_differences, size=sample_size, replace=False)
            r = np.sqrt(np.sum(np.cos(phase_differences))**2 + np.sum(np.sin(phase_differences))**2) / sample_size
            return 1 - r
        else:
            return np.nan

    def full_histograms(self, num_bins=20, subplots_per_row=6, figsize=(12, 4)):
        fig_num = 0
        for overlap_num, (overlapping_field_nums, overlap, phase_differences) in \
                enumerate(zip(self.overlapping_field_nums, self.overlaps, self.phase_differences)):
            if overlap_num % subplots_per_row == 0:
                fig, ax = plt.subplots(2, subplots_per_row, constrained_layout=True, figsize=figsize)
                ax[0, 0].set_ylabel('Firing rate (Hz)')
                ax[0, 0].set_xlabel('Position (cm)')
                ax[1, 0].set_ylabel('Count')
                ax[1, 0].set_xlabel('Pairwise phase\ndifferences (deg)')

            overlap = np.round(
                np.array(overlap - self.firing_fields.tracking.d_runs_offset) / self.firing_fields.bin_size).astype(int)
            overlap_range = slice(overlap[0], overlap[1])
            for overlapping_field_num in overlapping_field_nums:
                pair_num = self.fields['pair_nums'][overlapping_field_num]
                run_type = self.fields['run_types'][overlapping_field_num]
                firing_rate = self.firing_fields.smooth_rate_maps[pair_num, run_type, overlap_range]
                ax[0, overlap_num % subplots_per_row].plot(self.firing_fields.positions[overlap_range], firing_rate)
            if len(phase_differences):
                ax[1, overlap_num % subplots_per_row].hist(np.concatenate(phase_differences), bins=num_bins,
                                                           range=(0, 360))

            if overlap_num % subplots_per_row == subplots_per_row - 1 or overlap_num == len(self.overlaps) - 1:
                self.maybe_save_fig(fig, f"batch_{fig_num}")
                fig_num += 1

    def coordination_by_speed(self, speed_groups, num_phase_bins=20, sample_size=20, subplots_per_row=6,
                              fig_size=(12, 10)):
        fig_num = 0
        variances = []
        for overlap_num, (overlapping_field_nums, overlap, phase_differences, cycle_speeds) in \
                enumerate(zip(self.overlapping_field_nums, self.overlaps, self.phase_differences, self.cycle_speeds)):

            if overlap_num % subplots_per_row == 0:
                fig, ax = plt.subplots(len(speed_groups), subplots_per_row, sharex='all', figsize=fig_size)
                ax[-1, 0].set_ylabel('Count')
                ax[-1, 0].set_xlabel('Pairwise phase\ndifferences (deg)')

            groups_phase_differences = [[] for _ in range(len(speed_groups))]
            for cycle_speed, cycle_phase_differences in zip(cycle_speeds, phase_differences):
                for speed_group_num, speed_group in enumerate(speed_groups):
                    if speed_group[0] < cycle_speed < speed_group[1]:
                        groups_phase_differences[speed_group_num].append(cycle_phase_differences)
                    if speed_group[0] > cycle_speed:
                        break

            overlap_variances = []
            for speed_group_num, speed_group in enumerate(speed_groups):
                if len(groups_phase_differences[speed_group_num]):
                    axis = ax[speed_group_num, overlap_num % subplots_per_row]
                    group_phase_differences = np.concatenate(groups_phase_differences[speed_group_num])
                    axis.hist(group_phase_differences, bins=num_phase_bins, range=(0, 360))
                    v = self.subsampled_variance(group_phase_differences, sample_size=sample_size)
                    overlap_variances.append(v)
                    axis.annotate(f"v = {v:.2f}", (0.6, 0.85), xycoords='axes fraction', fontsize='x-small')
                else:
                    overlap_variances.append(np.nan)

                ax[speed_group_num, 0].annotate(f"{speed_group}\ncm/s", (-1, 0.5), xycoords="axes fraction",
                                                fontsize="large", rotation="vertical", va='center',
                                                multialignment="center")

            if overlap_num % subplots_per_row == subplots_per_row - 1 or overlap_num == len(self.overlaps) - 1:
                fig.tight_layout()
                self.maybe_save_fig(fig, f"batch_{fig_num}", subfolder="per_speed")
                fig_num += 1

            variances.append(overlap_variances)
        variances = np.array(variances)

        self.maybe_pickle_results(variances, name="variances", subfolder="per_speed")

        if (~np.isnan(variances)).any():
            fig, ax = plt.subplots(2, 1, sharex='col')
            average_speeds = np.mean(speed_groups, axis=1)
            for overlap_variances in variances:
                ax[0].plot(average_speeds, overlap_variances, '*-')
            ax[0].set_ylabel('Circular variance')

            clean_variances = []
            xs = []
            for group_num, group_variances in enumerate(variances.T):
                clean_overlap_variances = group_variances[~np.isnan(group_variances)]
                if len(clean_overlap_variances):
                    clean_variances.append(clean_overlap_variances)
                    xs.append(average_speeds[group_num])
            ax[1].axhline(np.nanmean(variances), linestyle='dotted', color='C7')
            ax[1].violinplot(clean_variances, xs, showmeans=True, showextrema=False,
                             widths=0.5 * (average_speeds[1] - average_speeds[0]))
            ax[1].set_ylabel('Circular variance')
            ax[1].set_xlabel('Running speed (cm/s)')

            self.maybe_save_fig(fig, "variances", subfolder="per_speed")

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):

        return cls(super_group_name, group_name, child_name, kwargs['LFP'], kwargs['FiringFields'],
                   parameters_dict['min_overlap'], parameters_dict['extra_margin'],
                   save_figures=save_figures, figure_format=figure_format, figures_path=figures_path,
                   pickle_results=pickle_results, pickles_path=pickles_path)
