import matplotlib.pyplot as plt
from data_analysis.analyze.config import figures_path, general_parameters, small_plots, cm
from data_analysis.analyze.batch_config import group_names
from data_analysis.batch import plot_pooled, plot_scatter, speed_histograms, cycle_lengths_vs_mean_x,\
    pooled_and_scatter_summary, characteristic_speeds, optimize_theta_times, speed_bin_ok_ratios, \
    phase_precession_per_acceleration, significant_cycles, scatter_x_vs_y, plot_histograms, pooled_slope_histogram, \
    within_field_increases


# By default, don't do any of the analyses
pooled_sizes = False
scatter_sizes = False
pooled_slopes = False
pooled_residuals = False
single_pass_slopes = False
scatter_slopes = False
pooled_lengths = False
pooled_lengths_restricted = False
single_cycle_lengths = False
variances = False
firing_rates = False
skewness = False
histograms = False
pooled_speeds = False
length_vs_size = False
pas_lengths = False
theta_times = False
ok_ratios = False
acceleration_clouds = False
curvatures = False
significant_ratios = False
size_vs_slope = False
kamran_histograms = False


# CHOOSE WHAT TO DO

close_figures = 1  # close figures after plotting and saving
analysis_of_variance = 0  # perform likelihood-ratio tests


# pas_lengths = True  # Fig 3B, D
pooled_lengths = True  # Fig 7
# pooled_lengths_restricted = True  # Fig 4 - fig sup 1
# single_cycle_lengths = True  # Fig 6D, G; Fig 6 - fig sup 1
#
pooled_sizes = True  # Fig 4B, C; Fig 7
# scatter_sizes = True  # Fig 6E, H; Fig 6 - fig sup 2A
# skewness = True  # Fig 7 - fig sup 1
#
pooled_slopes = True  # Fig 5B, C; Fig 7
# pooled_residuals = True  # Appendix 1 - fig 2A, B
# single_pass_slopes = True  # Fig 5E, F
# scatter_slopes = True  # Fig 6F, I & Fig 6 - fig sup 2B
#
# variances = True  # Appendix 1 - fig 2C, D
#
# firing_rates = True  # Fig 7 - fig sup 2
#
# histograms = True
# pooled_speeds = True  # Fig 6B
#
# ok_ratios = True  # Fig 6C
#
# size_vs_slope = True  # Fig 6 - fig sup 3A
# length_vs_size = True  # Fig 6 - fig sup 3B, C

# theta_times = True
#
# acceleration_clouds = True
# curvatures = True
#
# significant_ratios = True  # calculate percentages of significant theta cycles for each animal

# kamran_histograms = True


# define figure sizes
if small_plots:
    fig_sizes = ((17*cm, 5*cm), (17 * cm, 6.5 * cm), (17*cm, 10*cm))
    fig_summary_sizes = ((10.3*cm, 5.5*cm), (8 * cm, 8 * cm), (10*cm, 10*cm))
    fig_all_sizes = ((5.32*cm, 5.45*cm),
                     # (5.5 * cm, 9.95 * cm),
                     (5.3*cm, 9.9*cm),  # in the paper
                     (4.5*cm, 10*cm))
    fig_pooled_and_scatter_size = (6.6*cm, 8.8*cm)
    fig_violin_sizes = ((5.5*cm, 5.5*cm),)
else:
    fig_sizes = ((15, 5), (14, 7), (14, 8), (15, 12))
    fig_summary_sizes = ((7.2, 3.8), (10, 8), (10, 10), (10, 12))
    fig_violin_sizes = ((5.5, 3.5),)
    # fig_all_sizes = ((3.3, 3.3), (3.3, 6.5), (3.3, 9))
    fig_all_sizes = ((4, 4), (3.3, 6.5), (3.3, 9))
    fig_speed_histogram_size = (14, 3.5)
    fig_pooled_and_scatter_size = (3.8, 5.5)


# default parameters for moving averages
window_size = 15
window_stride = 3
window_min_points = 3


# figure for all violin plots
if pooled_sizes or pooled_slopes or pooled_lengths:
    fig_violins, ax_violins = plt.subplots(nrows=5 + 2*variances, ncols=len(group_names), sharey='row', sharex='row',
                                           squeeze=False, constrained_layout=True,
                                           # figsize=(len(group_names)*2*cm+2*cm, 13.5*cm+variances*3*cm),
                                           figsize=(5*cm, 12.5*cm))
    ax_violins[0, 0].set_ylim([0, 35])
    ax_violins[1, 0].set_ylim([20, 125])
    ax_violins[2, 0].set_ylim([-50, 0])
    ax_violins[3 + variances, 0].set_ylim([-50, 50])
    ax_violins[4 + variances, 0].set_ylim([-30, 20])

    if variances:
        ax_violins[5 + variances, 0].set_ylim([-0.25, 0.25])

    for col in range(len(group_names)):
        ax_violins[0, col].set_xticklabels([])
        ax_violins[1, col].set_xticklabels([])
        ax_violins[2 + variances, col].set_xlabel("Running speed\n(cm/s)")
        ax_violins[-2, col].set_xticklabels([])
        ax_violins[-1, col].set_xlabel("Deviation from\ncharacteristic\nspeed (cm/s)")

    if len(group_names) == 4:
        ax_violins[0, 1].set_title("Temporal\nsweep")
        ax_violins[0, 2].set_title("Spatial\nsweep")
        ax_violins[0, 3].set_title("Behavior-\ndependent\nsweep")

        for ax_indices in ((0, 1), (1, 1), (2, 1), (3, 2), (4, 2), (0, 3), (1, 3), (2, 3), (3, 3), (4, 3)):
            for key in ['top', 'bottom', 'left', 'right']:
                ax_violins[ax_indices].spines[key].set_color("#6ACC65")  # green
            ax_violins[ax_indices].tick_params(color="#6ACC65")
        for ax_indices in ((3, 1), (4, 1), (0, 2), (1, 2), (2, 2)):
            for key in ['top', 'bottom', 'left', 'right']:
                ax_violins[ax_indices].spines[key].set_color("#D65F5F")  # red
            ax_violins[ax_indices].tick_params(color="#D65F5F")

    elif len(group_names) == 2:
        for ax_indices in ((0, 1), (2, 1), (3, 1)):
            for key in ['top', 'bottom', 'left', 'right']:
                ax_violins[ax_indices].spines[key].set_color("#6ACC65")  # green
            ax_violins[ax_indices].tick_params(color="#6ACC65")
        for ax_indices in ((1, 1), (4, 1)):
            for key in ['top', 'bottom', 'left', 'right']:
                ax_violins[ax_indices].spines[key].set_color("#D65F5F")  # red
            ax_violins[ax_indices].tick_params(color="#D65F5F")

    for ax_row in ax_violins:
        for ax in ax_row:
            for key in ['top', 'bottom', 'left', 'right']:
                ax.spines[key].set_linewidth(0.75)
            for key in ['top', 'right']:
                ax.spines[key].set_visible(False)

x_labels = [
            # "Normalized run distance",
            "Distance from the\nstart of the run (cm)",
            # "Distance to the\nnearest border (cm)",
            # "Mean speed\nfor field's spikes (cm/s)",
            "Characteristic speed\nthrough the field (cm/s)"
            ]

x_labels_bis = ["Normalized run distance", "Characteristic speed\nthrough the field (cm/s)"]

little_names_x = ["distance",
                  # "spikes' speed",
                  "speed"]

if pooled_sizes:
    print("\nplotting place fields...")
    plot_pooled("1) Place field sizes pooled across speeds", ["Place field\nsize (cm)"],
                ["FiringFields/per_speed/field_sizes"], remove_outliers=True, inter_quartile_factor=3,
                violin_ax=ax_violins[1:2], violin_increments_ax=ax_violins[3+variances:4+variances],
                fig_size=fig_sizes[0], fig_summary_size=(5.2*cm, 10.3*cm), vertical_layout=True,
                close_figures=close_figures, slopes_x_label="Slope of place field size\nvs running speed (s)", p_pos=2,
                violin_increments_y_labels=[r"Within-field $\Delta$""\nplace field\nsize (cm)"],
                delta_delta=True, delta_delta_path="FiringFields/per_speed/size_increases")

if scatter_sizes:
    plot_scatter("2) Place field sizes", x_labels,
                 ["FiringFields/peak_distances_from_start",
                  # "FiringFields/peak_distances_to_border",
                  # "FiringFields/spikes_mean_speeds",
                  "FiringFields/characteristic_speeds"
                  ], ["Place field size\n(cm)"], ["FiringFields/sizes"],
                 window_sizes=[window_size, window_size], window_strides=[5, window_stride],
                 window_min_points=window_min_points, remove_outliers=True, inter_quartile_factor=3,
                 plot_slopes=[0, 1], all_together=[1, 1], all_together_means=[1, 1], all_together_dens=[0, 0],
                 analysis_of_variance=analysis_of_variance,
                 little_names_x=little_names_x, fig_size=fig_sizes[len(x_labels) - 1],
                 fig_summary_size=fig_summary_sizes[len(x_labels) - 1], fig_all_size=fig_all_sizes[len(x_labels) - 1],
                 close_figures=close_figures, annotation_xy=(0.05, 0.8))

    plot_scatter("2b) Place field sizes", x_labels_bis,
                 ["FiringFields/peak_normalized_pos",
                  # "FiringFields/peak_distances_to_border",
                  # "FiringFields/spikes_mean_speeds",
                  "FiringFields/characteristic_speeds"
                  ], ["Place field size (cm)"], ["FiringFields/sizes"],
                 window_sizes=[0.1, window_size], window_strides=[0.05, window_stride],
                 window_min_points=window_min_points, remove_outliers=True, inter_quartile_factor=3,
                 plot_slopes=[0, 1], all_together=[1, 1], all_together_means=[1, 1], all_together_dens=[0, 0],
                 analysis_of_variance=analysis_of_variance,
                 little_names_x=little_names_x, fig_size=fig_sizes[len(x_labels) - 1],
                 fig_summary_size=fig_summary_sizes[len(x_labels) - 1], fig_all_size=fig_all_sizes[len(x_labels) - 1],
                 close_figures=close_figures, annotation_xy=(0.05, 0.8))

    # plot_scatter("2c) Place field sizes",
    #              ["Distance to the\nnearest border (cm)", "Characteristic speed\nthrough the field (cm/s)"],
    #              ["FiringFields/peak_distances_to_border",
    #               # "FiringFields/spikes_mean_speeds",
    #               "FiringFields/characteristic_speeds"
    #               ], ["Place field size\n(cm)"], ["FiringFields/sizes"],
    #              window_sizes=[window_size, window_size], window_strides=[5, window_stride],
    #              window_min_points=window_min_points, remove_outliers=True, inter_quartile_factor=3,
    #              plot_slopes=[1, 1], all_together=[1, 1], all_together_means=[1, 1], all_together_dens=[0, 0],
    #              analysis_of_variance=analysis_of_variance,
    #              little_names_x=little_names_x, fig_size=fig_sizes[len(x_labels) - 1],
    #              fig_summary_size=fig_summary_sizes[len(x_labels) - 1], fig_all_size=fig_all_sizes[len(x_labels) - 1],
    #              close_figures=close_figures, annotation_xy=(0.05, 0.8))

if pooled_slopes:
    print("\nplotting phase precession...")
    plot_pooled("3) Phase precession pooled across speeds",
                ["Phase precession\nslope (??/cm)"], ["PhaseVsPosition/pooled/slopes"], remove_outliers=True,
                inter_quartile_factor=3, violin_ax=ax_violins[2:3],
                violin_increments_ax=ax_violins[4+variances:5+variances],
                fig_size=fig_sizes[0], fig_summary_size=fig_summary_sizes[0], close_figures=close_figures, p_pos=0.5,
                slopes_x_label="Slope of phase precession slope\n"r"vs running speed (??$\cdot$s)",
                violin_increments_y_labels=[r"Within-field $\Delta$""\nphase precession\nslope (??/cm)"],
                delta_delta=True, delta_delta_path="FiringFields/pooled/slope_increases")

if pooled_residuals:
    plot_pooled("3) Residuals of phase precession fits pooled across speeds",
                ["Mean squared residual (a.u.)"], ["PhaseVsPosition/pooled/mean_errors"], remove_outliers=True,
                inter_quartile_factor=3, fig_size=fig_sizes[0], fig_summary_size=(9*cm, 4.5*cm),
                close_figures=close_figures, p_pos=0.0005,
                slopes_x_label=r"$\Delta$ mean squared residual""\n"r"/ $\Delta$ running speed (s/cm)",
                violin_increments_y_labels=["Within-field\n"r"$\Delta$"" mean square error"],
                summary_y_lims=(0.01, 0.022))


if single_pass_slopes:
    plot_scatter("4) Single pass phase precession", ["Running speed (cm/s)"],
                 ["PhaseVsPosition/single_runs/speeds"],  ["Phase precession slope (??/cm)"],
                 ["PhaseVsPosition/single_runs/slopes"], window_sizes=[window_size],
                 window_strides=[window_stride], window_min_points=window_min_points,
                 remove_outliers=True, inter_quartile_factor=6, plot_slopes=[1], all_together=[1],
                 plot_violins=[1], fig_size=fig_sizes[0], fig_summary_size=fig_summary_sizes[0],
                 fig_all_size=fig_all_sizes[0], fig_violins_size=fig_all_sizes[0], close_figures=close_figures,
                 p_pos=0.8, slopes_x_lim=[-1.75, 1.75], shade=True,
                 slopes_x_label="Slope of phase precession slope\n"r"vs running speed (??$\cdot$s)")

if scatter_slopes:
    plot_scatter("5) Phase precession", x_labels,
                 ["PhaseVsPosition/all_spikes/peak_distances_from_start",
                  # "PhaseVsPosition/all_spikes/peak_distances_to_border",
                  # "PhaseVsPosition/all_spikes/spike_mean_speeds",
                  "PhaseVsPosition/all_spikes/characteristic_speeds"
                  ], ["Inverse phase precession\nslope (cm/??)"], ["PhaseVsPosition/all_spikes/slopes"],
                 # window_sizes=[0.1, 15, 15], window_strides=[0.05, 5, 5],
                 window_sizes=[window_size, window_size], window_strides=[5, window_stride],
                 window_min_points=window_min_points, remove_outliers=False,
                 inter_quartile_factor=3, plot_slopes=[0, 1], all_together=[1, 1], all_together_means=[1, 0],
                 all_together_dens=[0, 0], analysis_of_variance=analysis_of_variance, little_names_x=little_names_x,
                 fig_size=fig_sizes[len(x_labels) - 1], fig_summary_size=fig_summary_sizes[len(x_labels) - 1],
                 fig_all_size=fig_all_sizes[len(x_labels) - 1], close_figures=close_figures, hyperbolic_fit=[0, 1],
                 annotation_xy=(0.08, 0.1), inverse=True, y_lim=(-0.4, 0))

    plot_scatter("5b) Phase precession", x_labels_bis,
                 ["PhaseVsPosition/all_spikes/peak_normalized_pos",
                  # "PhaseVsPosition/all_spikes/peak_distances_to_border",
                  # "PhaseVsPosition/all_spikes/spike_mean_speeds",
                  "PhaseVsPosition/all_spikes/characteristic_speeds"
                  ], ["Inverse phase precession\nslope (cm/??)"], ["PhaseVsPosition/all_spikes/slopes"],
                 window_sizes=[0.1, window_size], window_strides=[0.05, window_stride], window_min_points=window_min_points,
                 remove_outliers=False, inter_quartile_factor=3, plot_slopes=[0, 1], all_together=[1, 1],
                 all_together_means=[1, 1], analysis_of_variance=analysis_of_variance, little_names_x=little_names_x,
                 fig_size=fig_sizes[len(x_labels) - 1], fig_summary_size=fig_summary_sizes[len(x_labels) - 1],
                 fig_all_size=(4.5*cm, 10.1*cm), close_figures=close_figures, hyperbolic_fit=[0, 0],
                 annotation_xy=(0.08, 0.1), inverse=True, y_lim=(-0.4, 0))

    # plot_scatter("5c) Phase precession",
    #              ["Distance to the\nnearest border (cm)", "Characteristic speed\nthrough the field (cm/s)"],
    #              ["PhaseVsPosition/all_spikes/peak_distances_to_border",
    #               # "PhaseVsPosition/all_spikes/spike_mean_speeds",
    #               "PhaseVsPosition/all_spikes/characteristic_speeds"
    #               ], ["Inverse phase precession\nslope (cm/??)"], ["PhaseVsPosition/all_spikes/slopes"],
    #              window_sizes=[0.1, window_size], window_strides=[0.05, window_stride], window_min_points=window_min_points,
    #              remove_outliers=False, inter_quartile_factor=3, plot_slopes=[1, 1], all_together=[1, 1],
    #              all_together_means=[1, 1], analysis_of_variance=analysis_of_variance, little_names_x=little_names_x,
    #              fig_size=fig_sizes[len(x_labels) - 1], fig_summary_size=fig_summary_sizes[len(x_labels) - 1],
    #              fig_all_size=(4.5*cm, 10.1*cm), close_figures=close_figures, hyperbolic_fit=[0, 0],
    #              annotation_xy=(0.08, 0.1), inverse=True, y_lim=(-0.4, 0))


if pooled_lengths:
    print("\nplotting theta trajectory length...")
    plot_pooled("6) Theta trajectory length - Whole track", ["Theta trajectory\nlength (cm)"],
                ["PathLengths/averaged_cycles/everywhere/path_lengths"], pairwise_increments=False,
                violin_ax=ax_violins[0:1], fig_size=fig_sizes[0], fig_summary_size=fig_summary_sizes[0],
                close_figures=close_figures, min_points=1)

if pooled_lengths_restricted:
    plot_pooled("6b) Theta trajectory length - Restricted", ["Theta trajectory length (cm)"],
                ["PathLengths/averaged_cycles/restricted/path_lengths"], pairwise_increments=False,
                fig_size=fig_sizes[0], fig_summary_size=fig_summary_sizes[0], fig_violins_size=fig_violin_sizes[0],
                close_figures=close_figures, min_points=1, summary_y_lims=(0, 35))

if single_cycle_lengths:
    hist_dicts = [{'x_min': 0, 'x_max': 1, 'num_x_bins': 25, 'y_min': -50, 'y_max': 75, 'num_y_bins': 26},
                  {'x_min': 0, 'x_max': 90, 'num_x_bins': 25, 'y_min': -50, 'y_max': 75, 'num_y_bins': 26}]

    plot_scatter("7) Single cycle theta trajectory lengths",
                 ["Normalized run distance",
                  # "Running speed (cm/s)",
                  "Characteristic running speed (cm/s)"],
                 ["PathLengths/single_cycles/normalized_pos",
                  # "PathLengths/single_cycles/speeds",
                  "PathLengths/single_cycles/characteristic_speeds"],
                 ["Theta trajectory\nlength (cm)"], ["PathLengths/single_cycles/path_lengths"],
                 window_sizes=[0.1, window_size], window_strides=[0.05, window_stride],
                 window_min_points=window_min_points, remove_outliers=True, inter_quartile_factor=6,
                 plot_slopes=[0, 1], all_together=[1, 1], all_together_means=[1, 1],
                 all_together_hists=[1, 1], fig_hist_size=fig_all_sizes[0],
                 hist_dicts=hist_dicts, plot_violins=[0, 0], shade=True,
                 fig_size=fig_sizes[1], fig_summary_size=fig_summary_sizes[1], fig_all_size=fig_all_sizes[1],
                 fig_violins_size=fig_all_sizes[1], close_figures=close_figures)

    # plot_histograms("7b) Single cycle theta trajectory length histograms",
    #                 ["Normalized run distance", "Characteristic\nrunning speed (cm/s)"],
    #                 ["PathLengths/single_cycles/normalized_pos", "PathLengths/single_cycles/characteristic_speeds"],
    #                 ["Theta trajectory length (cm)"], ["PathLengths/single_cycles/path_lengths"], plot_slopes=[0, 1],
    #                 hist_dicts=hist_dicts, min_points=10, fig_size=(17 * cm, 9 * cm), r_pos=(0.5, 0.08))

if variances:
    print("plotting phase difference variances...")
    plot_pooled("8) Variance in pairwise spike's phase differences",
                ["Variance in pairwise\nphase differences"],
                ['CellCoordination/per_speed/variances'], pairwise_increments=True,
                # violin_ax=ax_violins[3:4], violin_increments_ax=ax_violins[6:7],
                violin_increments_y_labels=["Within-pair\n"r"$\Delta$"" variance in spike's\nphase differences"],
                fig_size=fig_sizes[0], fig_summary_size=(9*cm, 4.5*cm), close_figures=close_figures, p_pos=0.01,
                slopes_x_label=r"$\Delta$ variance in pairwise phase""\n"
                               r"differences / $\Delta$ running speed (s/cm)",
                summary_y_lims=(0.45, 0.9))

if firing_rates:
    print("plotting peak firing rates...")
    plot_pooled("9) Peak firing rates", ["Peak firing rate (Hz)"], ['FiringFields/per_speed/field_peak_rates'],
                pairwise_increments=True, fig_size=fig_sizes[0], fig_summary_size=fig_summary_sizes[0],
                fig_violins_size=fig_violin_sizes[0], close_figures=close_figures,
                slopes_x_label="Slope of firing rate\n"r"vs running speed (cm$^{-1}$)", p_pos=0.3,
                summary_y_lims=(0, 25))

if skewness:
    plot_scatter("10) Skewness vs acceleration", [r'Mean acceleration (cm/s$^2$)'], ['FiringFields/accelerations'],
                 ['Field skewness'], ['FiringFields/skewnesses'], window_sizes=[0.6],
                 window_strides=[0.2], window_min_points=1, remove_outliers=False,
                 plot_slopes=[1], all_together=[1], fig_size=fig_sizes[0], fig_summary_size=fig_summary_sizes[0],
                 fig_all_size=fig_all_sizes[0], close_figures=close_figures, annotation_xy=(0.08, 0.8), y_lim=(-0.75, 1))

if histograms:
    speed_histograms(plot_colorbars=False, fig_size=fig_speed_histogram_size, close_figure=close_figures)

if pooled_speeds:
    characteristic_speeds("Characteristic speeds", "Tracking/characteristic_speeds",
                          general_parameters['Tracking']['spatial_bin_size'], alpha=1, fig_size=fig_sizes[0],
                          fig_summary_size=(5.32*cm, 5.18*cm), close_figures=close_figures)

if length_vs_size:
    cycle_lengths_vs_mean_x("FiringFields/peak_distances_from_start", "FiringFields/sizes", "place field size",
                            "PathLengths/single_cycles/distances_from_start", "PathLengths/single_cycles/path_lengths",
                            window_size=window_size, window_stride=window_stride, window_min_points=window_min_points,
                            fig_size=fig_sizes[0], close_figure=close_figures, r_pos=(0.7, 0.85))
    cycle_lengths_vs_mean_x("PhaseVsPosition/all_spikes/peak_distances_from_start", "PhaseVsPosition/all_spikes/slopes",
                            "inverse phase precession slope", "PathLengths/single_cycles/distances_from_start",
                            "PathLengths/single_cycles/path_lengths", window_size=window_size,
                            window_stride=window_stride, window_min_points=window_min_points, fig_size=fig_sizes[0],
                            close_figure=close_figures, inverse=True, units="cm/??", r_pos=(0.7, 0.85))

if pas_lengths:
    pooled_and_scatter_summary(fig_name="A - Summary of theta trajectory lengths.pdf",
                               pooled_names=["Pooled\n"r"$\theta$ trajectory length (cm)"],
                               pooled_paths=["PathLengths/averaged_cycles/everywhere/path_lengths"],
                               scatter_path_x="PathLengths/single_cycles/speeds",
                               scatter_names_y=["Single cycle\n"r"$\theta$ trajectory length (cm)"],
                               scatter_paths_y=["PathLengths/single_cycles/path_lengths"],
                               fig_size=fig_pooled_and_scatter_size, close_figure=close_figures)

if theta_times:
    optimize_theta_times()

if pooled_sizes or pooled_slopes or pooled_lengths:
    fig_violins.align_ylabels()
    fig_violins.savefig(f"{figures_path}/ALL/Violins", dpi=400, bbox_inches='tight')
    if close_figures:
        plt.close(fig_violins)

if ok_ratios:
    speed_bin_ok_ratios("FiringFields/ok_by_speed_distance", fig_size=(4.08*cm, 5.25*cm), close_figure=close_figures)

if acceleration_clouds:
    # acceleration_groups = ((-2, -0.6), (-0.6, -0.4), (-0.4, -0.2), (-0.2, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1))
    acceleration_groups = ((-1.5, -0.5), (-0.25, 0.25), (0.5, 1.5))
    phase_precession_per_acceleration("PhaseVsPosition/acceleration/positions", "PhaseVsPosition/acceleration/phases",
                                      "PhaseVsPosition/acceleration/speeds",
                                      "PhaseVsPosition/acceleration/accelerations",
                                      "PhaseVsPosition/acceleration/rel_c_speed_changes",
                                      "PhaseVsPosition/acceleration/circular_means",
                                      "PhaseVsPosition/acceleration/mean_ranges", acceleration_groups,
                                      num_position_bins=20, num_phase_bins=20)

if curvatures:
    plot_scatter("11) Curvature", ["Acceleration (cm/s^2)", "Speed change", "Relative speed change"],
                 ["PhaseVsPosition/acceleration/accelerations", "PhaseVsPosition/acceleration/c_speed_slopes",
                  "PhaseVsPosition/acceleration/rel_c_speed_changes"],
                 ["Curvature"], ["PhaseVsPosition/acceleration/curvatures"], window_sizes=[5, 0.5, 0.1],
                 window_strides=[2, 0.2, 0.01], window_min_points=2, remove_outliers=True, inter_quartile_factor=4,
                 plot_slopes=[1, 1, 1], all_together=[1, 1, 1], all_together_means=[0, 0, 0],
                 little_names_x=["acceleration", "c_speed_slope"], fig_size=fig_sizes[2],
                 fig_summary_size=fig_summary_sizes[2], fig_all_size=fig_all_sizes[2],
                 close_figures=close_figures, annotation_xy=(0.6, 0.8))

if significant_ratios:
    significant_cycles()

if size_vs_slope:
    scatter_x_vs_y("Place field size vs inverse phase precession slope", "Place field size (cm)",
                   "FiringFields/sizes", "FiringFields/indices", "Inverse\nphase precession slope (cm/??)",
                   "PhaseVsPosition/all_spikes/slopes", "PhaseVsPosition/all_spikes/indices", y_lim=-0.4,
                   fig_size=(5.32*cm, 4.9*cm))


if kamran_histograms:
    for num_bins in [21, 31, 41]:
        pooled_slope_histogram(f"K - Place field size increment histograms, {num_bins} bins",
                               r"$\Delta$ place field size / $\Delta$ running speed (s)",
                               "FiringFields/per_speed/field_sizes", remove_outliers=True, inter_quartile_factor=3,
                               close_figures=close_figures, num_bins=num_bins, fig_size=(6.5*cm, 4*cm))

        pooled_slope_histogram(f"K - Phase precession slope increment histograms, {num_bins} bins",
                               r"$\Delta$ phase precession slope / $\Delta$ running speed (deg$\cdot$s)",
                               "PhaseVsPosition/pooled/slopes", remove_outliers=True, inter_quartile_factor=3,
                               close_figures=close_figures, num_bins=num_bins, fig_size=(6.5*cm, 4*cm))

plt.show()
