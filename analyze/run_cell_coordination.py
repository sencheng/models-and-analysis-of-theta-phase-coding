from data_analysis.analyze.config import *
from data_analysis.initializer import initialize
from data_analysis.cell_coordination import CellCoordination


def run_cell_coordination(cell_coordination):
    print("Analyzing cell coordination...")
    p = general_parameters['CellCoordination']

    # cell_coordination.full_histograms()
    cell_coordination.coordination_by_speed(speed_groups, num_phase_bins=p['num_phase_bins'],
                                            sample_size=p['sample_size'])


if __name__ == '__main__':
    # session = "ec013.156"
    # session = "ec013.386"
    # session = "ec013.395"
    # session = "ec016.234"
    # session = "ec016.233"
    session = "ec014.468"
    # session = "ec014.639"
    # session = "Achilles.10252013"
    # session = "Buddy.06272013"

    group_name = experimental_group_name

    cell_coordination = initialize((CellCoordination,), session, group_name)['CellCoordination']
    run_cell_coordination(cell_coordination)
    plt.show()
