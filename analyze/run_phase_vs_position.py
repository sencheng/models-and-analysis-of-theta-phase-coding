from data_analysis.initializer import initialize
from data_analysis.analyze.config import *
from data_analysis.phase_vs_position import PhaseVsPosition


def run_phase_vs_position(phase_vs_position):
    print('Calculating phase precession slopes...')
    p = {**general_parameters['ALL'], **general_parameters['PhaseVsPosition']}

    # phase_vs_position.find_theta_0(min_shift=-60, max_shift=60, num_shifts=31)

    # # plot spikes within a place field
    # field_num = 0
    # fit_type = "simple_orthogonal"
    # positions, phases, speeds = phase_vs_position.pool(field_num)
    # run_type = fields['run_types'][field_num]
    # place_field_bounds = fields['bounds'][field_num]
    # fig, ax = plt.subplots()
    # phase_vs_position.fit_and_plot(positions, phases, run_type, place_field_bounds, True, fit_type, 0, ax, speeds)

    # pool all cells, splitting by run speeds
    print(0)
    phase_vs_position.pool_all(full_speed_groups=speed_groups, fit_type=p['fit_type'], min_spikes=p['pooled_min_spikes'],
                               pool_by_pass_speed=False, spike_speed_threshold=False, min_occupancy=p['min_occupancy'],
                               min_spread=p['min_spread'], plot_fits=True, plot_occupancy=True, fig_size=(18*cm, 13*cm),
                               # fields_per_plot=3, field_nums=(8, 20, 22), fig_size=(7.8*cm, 10.1*cm), constrained_layout=0
                               )

    # single passes for one field
    # field_num = 4
    # phase_vs_position.single_passes(field_num, p['pass_min_spikes'], p['pass_min_duration'], p['pass_min_spread'],
    #                                 p['pass_max_variation'], fit_type=p['fit_type'], plot_fits=True,
    #                                 fig_width=16.25*cm, fig_row_height=1*cm, fig_extra_height=0)

    # all single passes
    print(1)
    phase_vs_position.all_single_passes(p['pass_min_spikes'], p['pass_min_duration'], p['pass_min_spread'],
                                        p['pass_max_variation'], fit_type=p['fit_type'], plot_fits=True,
                                        plot_slopes=False)

    print(2)
    phase_vs_position.slopes_vs_stuff(min_spikes=p['pooled_min_spikes'], fit_type=p['fit_type'], plot_fits=True)
    #
    # print(3)
    # phase_vs_position.curvature_by_acceleration(num_means=3, plot=True)


if __name__ == '__main__':
    # session = "ec013.156"
    session = "ec013.454"  # fields 4 and 16 nice for single passes; also nice for comparing to variable noise model
    # session = "ec013.395"
    # session = "ec014.468"  # fields 8, 20, 22 nice for pooled at different speed bins
    # session = "ec014.639"
    # session = "ec016.233"
    # session = "ec016.269"
    # session = "Achilles.10252013"
    # session = "Buddy.06272013"
    # session = "Cicero.09172014"

    # group_name = experimental_group_name
    # group_name = "SpeedDepSharp"
    # group_name = "Time"
    # group_name = "VariableNoise3"
    group_name = "VariableNoiseVanilla"

    # with open(f"fields/{session}.{group_name}.pkl", 'rb') as fields_file:
    #     fields = pickle.load(fields_file)['complete_fields']

    phase_vs_position = initialize((PhaseVsPosition,), session, group_name)['PhaseVsPosition']
    run_phase_vs_position(phase_vs_position)
    plt.show()
