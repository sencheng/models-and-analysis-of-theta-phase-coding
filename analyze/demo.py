import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from data_analysis.lfp import LFP
from data_analysis.tracking import Tracking
from data_analysis.phase_vs_position import PhaseVsPosition
from data_analysis.analyze.config import general_parameters, figures_path, pickles_path
from data_analysis.spikes import UniformSpikes
from data_analysis.firing_fields import FiringFields
from cool_plots.color_maps import cm, c1


# CODE FOR FIG. 1; Comparison of theta trajectories, phase precession and place field sizes at two different speeds

tick_size = 2.5
thin_line_width = 0.5
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 6,
                     'mathtext.default': 'regular', 'axes.linewidth': thin_line_width,
                     'xtick.major.width': thin_line_width, 'xtick.major.size': tick_size,
                     'ytick.major.width': thin_line_width, 'ytick.major.size': tick_size,
                     'lines.linewidth': 1, 'lines.markersize': 1.5})

num_reps = 10
super_group_name = "Demo"
group_names = [
               "TimeDemo",  # temporal sweeps
               "PositionDemo"  # spatial sweeps
               ]
durations = [800, 1400]
# durations = [400, 400]
speeds = [15, 60]
fig_size = (2, 3.5)

for rep_num in range(num_reps):
    for group_name in group_names:

        if "Time" in group_name:
            path_arrow_color = "#D65F5F"
            field_arrow_color = "#D65F5F"
        else:
            path_arrow_color = "#a22a2a"
            field_arrow_color = 'k'

        fig, ax = plt.subplots(5, len(speeds), gridspec_kw={'height_ratios': [5.5, 1, 1, 3, 1]}, sharey='row',
                               squeeze=False, figsize=fig_size, constrained_layout=True)

        ax[0, 0].set_ylabel("Position (cm)", verticalalignment='top')
        ax[0, 0].set_ylim((70, 135))
        ax[2, 0].set_ylabel("'True'\nplace\nfield", verticalalignment='top')
        ax[3, 0].set_ylabel(r'$\theta$ phase (°)', verticalalignment='top')
        ax[4, 0].set_ylabel("F (Hz)", verticalalignment='top')
        ax[4, 0].set_ylim((0, 20))

        for ax_row in ax:
            for axis in ax_row:
                for spine in ['top', 'right']:
                    axis.spines[spine].set_visible(False)

        for col_num in range(2):
            ax[0, col_num].set_title(f"{speeds[col_num]} cm/s")
            ax[0, col_num].set_xlabel("Time (s)")
            ax[1, col_num].axis('off')
            ax[2, col_num].get_shared_x_axes().join(ax[2, col_num], ax[3, col_num], ax[4, col_num])
            ax[2, col_num].set_xlim([60, 140])
            ax[2, col_num].set_ylim([-0.1, 1.1])
            ax[2, col_num].set_yticks([])
            ax[2, col_num].set_xticklabels([])
            for spine in ['left', 'top', 'right']:
                ax[2, col_num].spines[spine].set_visible(False)
            ax[3, col_num].set_xticklabels([])
            ax[3, col_num].set_ylim([0, 360])
            ax[4, col_num].set_xlabel("Position (cm)")

        for row_num in range(5):
            ax[row_num, 0].yaxis.set_label_coords(-0.4, 0.5)

        highest_firing_rate = 0
        all_field_bounds = []

        for speed_num, speed in enumerate(speeds):
            # LFP
            p = general_parameters['LFP']
            theta_frequency = 8
            lfp = LFP(super_group_name, group_name, "LFP", filter_order=p['filter_order'],
                      bandpass_frequencies=p['bandpass_frequencies'], save_figures=False, figures_path=figures_path,
                      pickles_path=pickles_path)

            lfp.generate_constant_theta(duration=durations[speed_num], frequency=theta_frequency)
            lfp.finish_initialization()
            lfp.phase_from_hilbert_transform()
            lfp.find_significant_theta(amplitude_percentile=p['significance_percentile'], plot_steps=False,
                                       plot_histogram=False)
            lfp.find_cycle_boundaries()
            lfp.find_significant_cycles()
            lfp.clean()

            # Tracking
            p = general_parameters['Tracking']
            tracking = Tracking(super_group_name, group_name, "Tracking", spatial_bin_size=5,
                                save_figures=False, figures_path=figures_path, pickles_path=pickles_path)
            tracking.generate_step_trajectories(track_length=200, duration=durations[speed_num],
                                                inter_trial_duration=17.13, sections=((0, 200),), speeds=(speed,))
            tracking.calculate_speed_2D(p["speed_sigma"], plot=False)
            tracking.linear_fit(p['fitting_min_speed_ratio'])
            tracking.project()
            tracking.split_full_runs(p['runs_splitting_in_corner_sigma'], p['runs_splitting_out_of_corner_sigma'],
                                     min_speed=p['runs_splitting_min_speed'], corner_sizes=p['corner_sizes'],
                                     plot_steps=False)
            tracking.calculate_speed_1D(p["speed_sigma"], plot=False)
            tracking.calculate_characteristic_speeds(top_percentile=p['top_percentile'],
                                                     bottom_speed_from=p['bottom_speed_from'],
                                                     median=p['median'])

            # Model Spikes
            p = general_parameters[f'UniformSpikes|{group_name}']
            num_cells = 3
            model_spikes = UniformSpikes(super_group_name, group_name, "Spikes", lfp, tracking, num_cells,
                                         p["field_sigma"], p["ds"], p["dt"], p["phase_range"], p["phase_current"],
                                         p["theta_time"], p["theta_distance"], p["firing_rate_0"],
                                         p["firing_rate_slope"], p["theta_modulation"], save_figures=False)
            model_spikes.generate_spikes()

            # Firing Fields
            p = general_parameters['FiringFields']
            firing_fields = FiringFields(super_group_name, group_name, "FiringFields", model_spikes.spikes, tracking,
                                         p['firing_rate_sigma'], p['consecutive_nans_max'], save_figures=False)
            firing_fields.find_fields_candidates(p['min_spikes'], p['min_peak_firing_rate'],
                                                 p['firing_rate_threshold'], p['peak_prominence_threshold'])
            firing_fields.accept_all()

            # PLOT
            ax[2, speed_num].plot(np.arange(model_spikes.num_spatial_bins)*model_spikes.ds,
                                  model_spikes.fields[0, :, 1])

            ax[4, speed_num].plot(firing_fields.positions, firing_fields.smooth_rate_maps[1, 0])
            field_bounds = firing_fields.cand_bounds[1]
            all_field_bounds.append(field_bounds)
            ax[4, speed_num].axvline(field_bounds[0], color='C7', linestyle='dotted')
            ax[4, speed_num].axvline(field_bounds[1], color='C7', linestyle='dotted')
            max_firing_rate = np.nanmax(firing_fields.smooth_rate_maps[1, 0])
            if max_firing_rate > highest_firing_rate:
                highest_firing_rate = max_firing_rate

            bound_indices = firing_fields.cand_bound_indices[1]
            field_size = firing_fields.positions[bound_indices[1]] - firing_fields.positions[bound_indices[0]]

            positions = []
            phases = []
            for spike_time in model_spikes.spikes.spike_times[1]:
                position, spike_run_type = tracking.at_time(spike_time)
                if spike_run_type == 0:
                    phase, = lfp.at_time(spike_time, 0, return_phase=True)
                    positions.append(position)
                    phases.append(phase)

            # Phase Vs Position
            p = general_parameters['PhaseVsPosition']
            phase_vs_position = PhaseVsPosition(super_group_name, group_name, "PhaseVsPosition", lfp, firing_fields,
                                                False, 0, 0,
                                                p['normalized_slope_bounds'], p['pass_min_speed'],
                                                p['pass_speeds_from'], p['orthogonal_fit_params'])
            slope, intercept, _ = phase_vs_position.simple_orthogonal_fit(np.copy(positions), np.copy(phases), field_size)
            ax[3, speed_num].scatter(positions, phases, c=np.array(phases) / 360, vmin=0, vmax=1, cmap=cm)
            hy = 320
            hx = (hy - intercept) / slope
            dx = 10
            ax[3, speed_num].plot((hx, hx+dx, hx+dx), (hy, hy, slope*(hx+dx)+intercept), solid_capstyle='round',
                                  color=field_arrow_color)
            ax[3, speed_num].annotate(f"{slope:.1f}", (hx + 16, hy), color=field_arrow_color,
                                      horizontalalignment='left', verticalalignment='top')
            x = np.array((field_bounds[0], field_bounds[1]))
            y = x * slope + intercept
            ax[3, speed_num].plot(x, y, color='C7')

            num_cycles = 3
            central_time = tracking.times[np.argmax(tracking.d > 100)]
            cycle_index = np.searchsorted(lfp.times[lfp.cycle_boundaries[0][:, 0]], central_time) - 1
            time_start = lfp.times[lfp.cycle_boundaries[0][cycle_index - num_cycles // 2, 0]]
            time_end = lfp.times[lfp.cycle_boundaries[0][cycle_index + num_cycles // 2 + 1, 0]]
            index_start = np.searchsorted(model_spikes.times, time_start)
            index_end = np.searchsorted(model_spikes.times, time_end)

            theta_paths = model_spikes.theta_paths[index_start:index_end]
            jumps = np.diff(theta_paths) < -5
            theta_paths[:-1][jumps] = np.nan
            c = np.zeros(len(theta_paths))
            jump_indices = np.where(jumps)[0]
            if jump_indices[0] > 1 / theta_frequency / model_spikes.dt / 2:
                jump_indices = np.append(0, jump_indices)
            if jump_indices[-1] < 1 / theta_frequency / model_spikes.dt * (num_cycles - 0.5):
                jump_indices = np.append(jump_indices, len(theta_paths))
            for cycle_start, cycle_end in zip(jump_indices[:-1], jump_indices[1:]):
                c[cycle_start+1:cycle_end] = np.linspace(0, 1, cycle_end-cycle_start-1)

            times = model_spikes.times[index_start:index_end]
            times -= times[0]
            ax[0, speed_num].plot(times, model_spikes.positions[index_start:index_end], 'k', zorder=1)
            ax[0, speed_num].plot(times, theta_paths, linewidth=0)

            points = np.array([times, theta_paths]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, 1)
            lc = LineCollection(segments, cmap=cm, norm=norm)
            lc.set_array(c)
            ax[0, speed_num].add_collection(lc)

            middle_index = int(round(((jump_indices[-2] + jump_indices[-1])/2)))
            arrow_bottom = theta_paths[jump_indices[-2]+1]
            arrow_top = theta_paths[jump_indices[-1]-1]
            ax[0, speed_num].plot((times[jump_indices[-2]+1], times[jump_indices[-1]-1]), (arrow_bottom, arrow_bottom),
                                  color=path_arrow_color)
            ax[0, speed_num].plot((times[jump_indices[-2]+1], times[jump_indices[-1]-1]), (arrow_top, arrow_top),
                                  color=path_arrow_color)

            p = general_parameters[f'UniformSpikes|{group_name}']
            length = speed * (p["theta_time"] + 1/theta_frequency) + p["theta_distance"]

            ax[0, speed_num].annotate('', (times[middle_index], arrow_bottom), xytext=(times[middle_index], arrow_top),
                                      arrowprops=dict(arrowstyle='<|-|>', shrinkA=0, shrinkB=0,
                                                      facecolor=path_arrow_color, edgecolor=path_arrow_color))
            ax[0, speed_num].annotate(f"{length:.1f}", (times[middle_index], arrow_top + 2),
                                      color=path_arrow_color, horizontalalignment='center')

            if speed_num == 1:
                l1 = Line2D([], [], color='k')
                l2 = Line2D([], [], color=c1)
                legend = ax[0, speed_num].legend([l1, l2], ["x(t)", 'r(t)'],
                                                 loc="lower right", fontsize="small")
                legend.get_frame().set_linewidth(thin_line_width)

        for speed_num in range(len(speeds)):
            ax[4, speed_num].set_ylim([0, 1.5*highest_firing_rate])
            arrow_y = 1.25*highest_firing_rate
            field_bounds = all_field_bounds[speed_num]
            ax[4, speed_num].annotate('', (field_bounds[0], arrow_y), xytext=(field_bounds[1], arrow_y),
                                      arrowprops=dict(arrowstyle='<|-|>', shrinkA=0, shrinkB=0,
                                                      facecolor=field_arrow_color, edgecolor=field_arrow_color))
            ax[4, speed_num].annotate(f"{field_bounds[1]-field_bounds[0]:.0f}",
                                      ((field_bounds[0] + field_bounds[1])/2, highest_firing_rate*1.4),
                                      color=field_arrow_color, horizontalalignment='center')

        folder = f"{figures_path}/Demo/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        fig.savefig(f"{folder}{group_name}_{rep_num}.pdf", transparent=True)

# plt.show()
