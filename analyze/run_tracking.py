import json
import matplotlib.pyplot as plt
from data_analysis.lfp import LFP
from data_analysis.tracking import Tracking
from data_analysis.analyze.config import data_path, general_parameters, experimental_group_name, figures_path, save_figures
from data_analysis.initializer import initialize


def run_tracking(tracking):
    print("\nCalculating tracking...")
    p = general_parameters['Tracking']
    tracking.plot_positions(plot_front_LED=False, plot_back_LED=False)
    tracking.plot_displacement()
    tracking.speed_vs_position(p['speed_bin_size'], fig_size=(6, 3))
    tracking.acceleration_vs_position(p['acceleration_bin_size'])
    # tuned for "ec014.639":
    tracking.speeds_sizes_sketch(p['speed_bin_size'], ds=(25, 200), ls=(0.25, 0.9), sigmas=(6, 20),
                                 rectangle_y_pos=((0, 60), (60, 105)), run_type=0)


if __name__ == '__main__':

    # session = "ec013.388"
    # session = "ec013.395"
    # session = "ec013.412"
    # session = "ec014.468"
    # session = "ec014.639"
    # session = "ec016.233"

    session = "vvp01.1815"
    # session = "gor01.1216"

    # session = "Achilles.10252013_alt"
    # session = "Buddy.06272013"
    # session = "Cicero.09012014"
    # session = "Cicero.09172014"
    # session = "Gatsby.08022013"

    with open('sessions/' + session + '.json') as session_file:
        s = json.load(session_file)

    p = {**general_parameters['Tracking'], **s}

    lfp = initialize((LFP,), session, experimental_group_name)['LFP']

    tracking = Tracking(session, experimental_group_name, "Tracking",
                        p['spatial_bin_size'], save_figures=save_figures, figures_path=figures_path, figure_format='pdf')
    tracking.load_tracking(data_path, s['dataset'], s['session_set'], s['session'], lfp, s['discarded_intervals'],
                           p['back_to_front_progress'], p['sampling_rate'])
    tracking.calculate_speed_2D(p["speed_sigma"], plot=False)
    tracking.linear_fit(p['fitting_min_speed_ratio'])
    tracking.project()
    tracking.split_full_runs(p['runs_splitting_in_corner_sigma'], p['runs_splitting_out_of_corner_sigma'],
                             min_speed=p['runs_splitting_min_speed'], corner_sizes=p['corner_sizes'], plot_steps=True)
    tracking.calculate_speed_1D(p["speed_sigma"], plot=True)
    tracking.calculate_characteristic_speeds(top_percentile=p['top_percentile'],
                                             bottom_speed_from=p['bottom_speed_from'], median=p['median'])

    run_tracking(tracking)

    # # print the displacement at some time as an example
    time = 100
    d, run_type = tracking.at_time(time)
    # print(f"At time {time} s, d: {d}, run_type: {run_type}, run_num: {run_num}")

    plt.show()
