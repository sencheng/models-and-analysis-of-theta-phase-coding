from data_analysis.analyze.config import *
from data_analysis.initializer import initialize
from data_analysis.lfp import LFP
from data_analysis.decoder import Decoder
from data_analysis.path_lengths import PathLengths


def decode_time_bins(decoder):
    lfp = initialize((LFP,), session, experimental_group_name)['LFP']
    decoder.decode_time_bins(bin_size=0.01, step_size=0.002, time_interval=[1174, 1180], min_spikes=1,
                             plot_most_probable_positions=True, plot_theta_peaks=True, lfp=lfp)


def decode_phase_bins(decoder):
    p = general_parameters['Decoder']
    lfp = initialize((LFP,), session, experimental_group_name)['LFP']
    decoder.decode_phase_bins(lfp, phase_bin_size=p['phase_bin_size'], phase_step_size=p['phase_step_size'],
                              min_spikes=p['min_spikes'], plot_decoded_probabilities=True, time_interval=(90, 100))


def run_path_lengths(path_lengths):
    print('Calculating path lengths...')
    p = {**general_parameters['ALL'], **general_parameters['PathLengths']}

    path_lengths.calculate_accepted_bins(speed_groups, p['min_occupancy'], p['min_spread'])

    path_lengths.averaged_path_lengths(p['group_names'], speed_groups, p['margins_lists'], p['restricted_occupancies'],
                                       min_cycles=p['min_cycles'], path_decoding=p['path_decoding'],
                                       hanning_width=p['hanning_width'], radon_fit_params=p['radon_fit_params'],
                                       speed_groups_to_plot=p['speed_groups_to_plot'], cycles_fig_size=(9.83*cm, 3.4*cm))

    path_lengths.single_cycles(min_peak_prob=p['min_peak_prob'], min_phase_coverage=p['min_phase_coverage'],
                               min_phase_extent=p['min_phase_extent'], radon_fit_params=p['radon_fit_params'],
                               max_cycles_to_plot=128, from_run_types=(1, 1), cycles_per_figure=(8, 8),
                               cycles_fig_size=(9.7*cm, 9*cm))

    # if path_lengths.group_name == experimental_group_name:
    #     # print("NO displacement compensation:")
    #     # print("Average speed predictions:")
    #     # theta_time = path_lengths.optimize_theta_times(bounds=(0.4, 0.8), num_points=15, model_type="speed",
    #     #                                                displacement_compensation=False)
    #     # slope, intercept, r = path_lengths.decoded_vs_average_speed_predictions(theta_time=theta_time, plot=True,
    #     #                                                                         displacement_compensation=False)
    #     # print(f"theta_time = {theta_time} -> slope = {slope}, intercept = {intercept}, r = {r}")
    #     #
    #     # print("Average time predictions:")
    #     # theta_time = path_lengths.optimize_theta_times(bounds=(0.4, 0.8), num_points=15, model_type="time",
    #     #                                                displacement_compensation=False)
    #     # slope, intercept, r = path_lengths.decoded_vs_average_time_predictions(theta_time=theta_time, plot=True,
    #     #                                                                        displacement_compensation=False)
    #     # print(f"theta_time = {theta_time} -> slope = {slope}, intercept = {intercept}, r = {r}")
    #
    #
    #     print("\nWith displacement compensation:")
    #     print("Average speed predictions:")
    #     theta_time = path_lengths.optimize_theta_times(bounds=(0.2, 0.7), num_points=20, model_type="speed")
    #     slope, intercept, r = path_lengths.decoded_vs_average_speed_predictions(theta_time=theta_time, plot=True)
    #     print(f"theta_time = {theta_time} -> slope = {slope}, intercept = {intercept}, r = {r}")
    #
    #     # print("Average time predictions:")
    #     # theta_time = path_lengths.optimize_theta_times(bounds=(0.2, 0.7), num_points=20, model_type="time")
    #     # slope, intercept, r = path_lengths.decoded_vs_average_time_predictions(theta_time=theta_time, plot=True)
    #     # print(f"theta_time = {theta_time} -> slope = {slope}, intercept = {intercept}, r = {r}")

    plt.show()


if __name__ == '__main__':
    # session = "ec013.206"
    # session = "ec013.412"
    session = "ec014.468"
    # session = "ec014.639"
    # session = "ec016.233"
    # session = "ec016.269"
    # session = "Buddy_06272013"
    # session = "Achilles.10252013"
    # session = "Gatsby.08022013"
    # session = "vvp01.1815"
    # session = "gor01.1215"

    group_name = experimental_group_name
    # group_name = "VariableNoiseFixed"

    objects = initialize((Decoder, PathLengths), session, group_name)
    # decode_time_bins(objects['Decoder'])
    # decode_phase_bins(objects['Decoder'])
    run_path_lengths(objects['PathLengths'])
    plt.show()
