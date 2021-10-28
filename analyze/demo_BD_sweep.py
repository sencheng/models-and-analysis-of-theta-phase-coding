from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from cool_plots.color_maps import cm as c_map
from cool_plots.color_maps import c1
from data_analysis.lfp import LFP
from data_analysis.tracking import Tracking
from data_analysis.analyze.config import general_parameters, figures_path, pickles_path
from data_analysis.spikes import SpeedSpikes
from data_analysis.firing_fields import FiringFields


# FIG 2; Demo of the behavior-dependent sweep model when changing instantaneous and characteristic running speeds

bottom = 110
top = 290
num_cycles = 3
theta_frequency = 8
duration = num_cycles / theta_frequency
speeds = (15, 60)
theta_distances = (7.5, 30)
central_positions = (250, 150)
points_per_cycle = 100
arrow_color = "#D65F5F"
colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]

inset_width = 0.45
inset_height = 80


tick_size = 2.0
thin_line_width = 0.5
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Arial', 'font.size': 6,
                     'mathtext.default': 'regular', 'axes.linewidth': thin_line_width,
                     'xtick.major.width': thin_line_width, 'xtick.major.size': tick_size,
                     'ytick.major.width': thin_line_width, 'ytick.major.size': tick_size,
                     'lines.linewidth': 1, 'lines.markersize': 1.5})

cm = 1/2.54
fig, ax = plt.subplots(1, 2, sharey='row', gridspec_kw={'width_ratios': [1, 0.2]}, figsize=(6.5*cm, 6*cm))
ax[0].set_visible(False)
# ax[1].invert_xaxis()
# ax[1].axis('off')
for spine in ['top', 'bottom', 'right']:
    ax[1].spines[spine].set_visible(False)
ax[1].set_xlim([0, 1.2])
# rect = plt.Rectangle((-0.2, bottom), 0.2, top - bottom, facecolor='whitesmoke')
# ax[1].add_patch(rect)
ax[1].set_ylim([bottom, top])
ax[1].set_xlabel("Firing\nrate")


def add_inset(left, bottom, width, height, x_label=None, y_label=None):
    ax_ins = inset_axes(ax[0], width="100%", height="100%", bbox_to_anchor=(left, bottom, width, height),
                        bbox_transform=ax[0].transData, loc='lower left', borderpad=0)
    ax_ins.set_ylim([bottom, bottom+height])

    for spine in ['top', 'right']:
        ax_ins.spines[spine].set_visible(False)
    if x_label is not None:
        ax_ins.set_xlabel(x_label)
    else:
        ax_ins.set_xticklabels([])
    if y_label is not None:
        ax_ins.set_ylabel(y_label)
    else:
        ax_ins.set_yticklabels([])

    center = bottom + height / 2
    ax_ins.yaxis.set_major_locator(ticker.FixedLocator(([center - 25, center, center + 25])))
    return ax_ins


ax_tl = add_inset(0, top-inset_height, inset_width, inset_height, y_label="Position (cm)")
ax_tr = add_inset(1 - inset_width, top-inset_height, inset_width, inset_height)
ax_bl = add_inset(0, bottom, inset_width, inset_height, y_label="Position (cm)", x_label="Time (s)")
ax_br = add_inset(1 - inset_width, bottom, inset_width, inset_height, x_label="Time (s)")
ax_ins = [[ax_tl, ax_tr], [ax_bl, ax_br]]

l1 = Line2D([], [], color='k')
l2 = Line2D([], [], color=c1)
legend = ax_tl.legend([l1, l2], ["x(t)", 'r(t)'], loc="lower left", fontsize='small')
legend.get_frame().set_linewidth(0.75)


for speed_num, speed in enumerate(speeds):
    x_inc = speed * duration
    for theta_distance, central_position, ax_row in zip(theta_distances, central_positions, ax_ins):
        ax_row[speed_num].plot((0, duration), (central_position - x_inc/2, central_position + x_inc/2), 'k', zorder=1,
                               linewidth=1.5)

        r_inc = theta_distance + speed/theta_frequency
        for cycle_num in range(num_cycles):
            x_center = (cycle_num + 0.5)/theta_frequency * speed + central_position - x_inc/2

            times = np.linspace(cycle_num/theta_frequency, (cycle_num+1)/theta_frequency, points_per_cycle)
            r = np.linspace(x_center - r_inc/2, x_center + r_inc/2, points_per_cycle)
            c = np.linspace(0, 1, points_per_cycle)

            points = np.array([times, r]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            norm = plt.Normalize(0, 1)
            lc = LineCollection(segments, cmap=c_map, norm=norm, linewidths=1.5)
            lc.set_array(c)
            ax_row[speed_num].add_collection(lc)

        t_bars = ((num_cycles - 1)/8, num_cycles/8)
        arrow_bottom = x_center - r_inc/2
        arrow_top = x_center + r_inc/2
        ax_row[speed_num].plot(t_bars, (arrow_bottom, arrow_bottom), color=arrow_color)
        ax_row[speed_num].plot(t_bars, (arrow_top, arrow_top), color=arrow_color)

        t_arrow = (num_cycles - 0.5) / 8
        ax_row[speed_num].annotate('', (t_arrow, arrow_bottom), xytext=(t_arrow, arrow_top),
                                   arrowprops=dict(arrowstyle='<|-|>', shrinkA=0, shrinkB=0,
                                                   facecolor=arrow_color, edgecolor=arrow_color))
        ax_row[speed_num].annotate(f"{r_inc:.1f}", (t_arrow, arrow_top + 3),
                                   color=arrow_color, horizontalalignment='center')


# FIRING FIELDS FROM THE MODEL
super_group_name = "BDSweepDemo"
group_name = "BDSweepDemoRevision"
t_sim = 1400

# LFP
p = general_parameters['LFP']
theta_frequency = 8
lfp = LFP(super_group_name, group_name, "LFP", filter_order=p['filter_order'],
          bandpass_frequencies=p['bandpass_frequencies'], save_figures=False, figures_path=figures_path,
          pickles_path=pickles_path)

lfp.generate_constant_theta(duration=t_sim, frequency=theta_frequency)
lfp.finish_initialization()
lfp.phase_from_hilbert_transform()
lfp.find_significant_theta(amplitude_percentile=p['significance_percentile'], plot_steps=False,
                           plot_histogram=False)
lfp.find_cycle_boundaries()
lfp.find_significant_cycles()
lfp.clean()

# Tracking
p = general_parameters['Tracking']
tracking = Tracking(super_group_name, group_name, "Tracking", spatial_bin_size=2.5,
                    save_figures=False, figures_path=figures_path)
tracking.generate_step_trajectories(track_length=290, duration=t_sim, inter_trial_duration=20,
                                    sections=((0, 200), (200, 290)), speeds=(60, 15))
tracking.calculate_speed_2D(p["speed_sigma"], plot=False)
tracking.linear_fit(p['fitting_min_speed_ratio'])
tracking.project()
tracking.split_full_runs(p['runs_splitting_in_corner_sigma'], p['runs_splitting_out_of_corner_sigma'],
                         min_speed=p['runs_splitting_min_speed'], corner_sizes=p['corner_sizes'], plot_steps=False)
tracking.calculate_speed_1D(p["speed_sigma"], plot=False)
tracking.calculate_characteristic_speeds(top_percentile=p['top_percentile'], bottom_speed_from=p['bottom_speed_from'],
                                         median=p['median'], min_speed_count_percentile=0)

# Model Spikes
p = general_parameters[f'SpeedSpikes|{group_name}']
num_cells = 2
# field_centers = [c - bottom + 100 for c in [138, 150, 162, 238, 250, 262]]
field_centers = [150, 250]

model_spikes = SpeedSpikes(super_group_name, group_name, "Spikes", lfp, tracking, num_cells, p["ds"], p["dt"],
                           p["phase_range"], p["phase_current"], p["firing_rate_0"], p["firing_rate_slope"],
                           p["theta_modulation"], p['theta_time'], p['multiplicative_sigma'],
                           p['additive_sigma'], p['size_to_theta_d'], p['size_min'], p['shift_sigma'], field_centers,
                           save_figures=False)
model_spikes.generate_spikes()

# Firing Fields
p = general_parameters['FiringFields']
firing_fields = FiringFields(super_group_name, group_name, "FiringFields", model_spikes.spikes, tracking,
                             p['firing_rate_sigma'], p['consecutive_nans_max'], save_figures=False)
firing_fields.find_fields_candidates(p['min_spikes'], p['min_peak_firing_rate'], p['firing_rate_threshold'],
                                     p['peak_prominence_threshold'])

path_arrow_color = "#D65F5F"
field_arrow_color = "#D65F5F"

for field_num, firing_rates in enumerate(firing_fields.smooth_rate_maps):
    normalized_rates = firing_rates[0] / np.nanmax(firing_rates[0])
    ax[1].plot(normalized_rates, firing_fields.positions, color='C0')
    field_bounds = firing_fields.cand_bounds[field_num*2]
    ax[1].axhline(field_bounds[0], color='C7', linestyle='dotted', zorder=0)
    ax[1].axhline(field_bounds[1], color='C7', linestyle='dotted', zorder=0)

    arrow_x = 1.1
    ax[1].annotate('', (arrow_x, field_bounds[0]), xytext=(arrow_x, field_bounds[1]),
                   arrowprops=dict(arrowstyle='<|-|>', shrinkA=0, shrinkB=0,
                                   facecolor=field_arrow_color, edgecolor=field_arrow_color))
    ax[1].annotate(f"{field_bounds[1] - field_bounds[0]:.0f}",
                   (1.2, (field_bounds[0] + field_bounds[1]) / 2),
                   color=field_arrow_color, verticalalignment='center')

# ax[1].axvline(-0.2, color='gray')
# ax[1].axvline(0, color='gray')
ax[1].set_xticks([])
ax[1].set_yticks([125, 150, 175, 200, 225, 250, 275])

# fig.tight_layout(pad=2, w_pad=5)
fig.savefig(f"figures/BD_sweep {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.pdf", bbox_inches='tight')
plt.show()
