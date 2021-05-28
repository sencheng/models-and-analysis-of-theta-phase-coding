from data_analysis.initializer import initialize
from data_analysis.analyze.config import *
from data_analysis.firing_fields import FiringFields


def run_firing_fields(firing_fields):
    print("Calculating place fields...")
    p = {**general_parameters['ALL'], **general_parameters['FiringFields']}

    # firing_fields.plot_traces(field_nums=(0, 1, 2))
    firing_fields.plot_heatmap()

    firing_fields.field_sizes_by_speed(speed_groups=speed_groups, min_peak_firing_rate=p['min_peak_firing_rate'],
                                       threshold=p['firing_rate_threshold'],
                                       peak_prominence_threshold=p['peak_prominence_threshold'],
                                       min_occupancy=p['min_occupancy'],
                                       min_spread=p['min_spread'],
                                       plot_fields=False,
                                       # plot_fields=True, field_nums=(22, 9, 10), fields_per_plot=3,
                                       # fig_size=(6*cm, 10*cm), constrained_layout=True
                                       )

    firing_fields.field_sizes_vs_stuff(plot=True)

    firing_fields.field_skewness_vs_acceleration()

    firing_fields.ok_speed_bins_histogram(speed_groups, p['min_occupancy'], p['min_spread'], hist_bin_size=10,
                                          num_hist_bins=6, verbose=True)


if __name__ == '__main__':
    # session = "ec013.156"
    # session = "ec013.412"
    # session = "ec013.395"
    session = "ec014.468"  # fields 22, 9, 10 nice for pooled at different speed bins
    # session = "ec014.639"
    # session = "ec016.233"
    # session = "ec016.269"
    # session = "2006-6-12_15-55-31"  # gor01
    # session = "2006-4-10_21-2-40"  # vvp01
    # session = "Achilles.10252013"
    # session = "Buddy.06272013"
    # session = "gor01.1215"
    # session = "vvp01.1815"

    firing_fields = initialize((FiringFields,), session, experimental_group_name)['FiringFields']
    run_firing_fields(firing_fields)
    plt.show()
