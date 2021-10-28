import os
import json
from data_analysis.initializer import initialize
from data_analysis.analyze.batch_config import sessions, group_names, experimental_group_name
from data_analysis.analyze.run_tracking import run_tracking, Tracking
from data_analysis.lfp import LFP
from data_analysis.analyze.run_firing_fields import run_firing_fields, FiringFields
from data_analysis.analyze.run_phase_vs_position import run_phase_vs_position, PhaseVsPosition
from data_analysis.analyze.run_decoder import run_path_lengths, PathLengths
from data_analysis.analyze.run_cell_coordination import run_cell_coordination, CellCoordination


def analyze(session, group_name):

    # if group_name == experimental_group_name:
    #     tracking = initialize((Tracking,), session, group_name)['Tracking']
    #     run_tracking(tracking)
    #     del tracking

    # place field sizes and skews
    firing_fields = initialize((FiringFields,), session, group_name)['FiringFields']
    run_firing_fields(firing_fields)
    del firing_fields

    # phase precession slopes
    phase_vs_position = initialize((PhaseVsPosition,), session, group_name)['PhaseVsPosition']
    run_phase_vs_position(phase_vs_position)
    del phase_vs_position

    # theta trajectory lengths
    path_lengths = initialize((PathLengths,), session, group_name)['PathLengths']
    run_path_lengths(path_lengths)
    del path_lengths

    # variance in pairwise spike phases
    cell_coordination = initialize((CellCoordination,), session, group_name)['CellCoordination']
    run_cell_coordination(cell_coordination)
    del cell_coordination


for session in sessions:
    # check that there are fields defined for all sessions
    for group_name in group_names:
        if not os.path.exists(f"fields/{session}.{group_name}.json"):
            firing_fields = initialize((FiringFields,), session, group_name)
            del firing_fields
    # check that a best phase shift has been calculated
    with open(f"sessions/{session}.json", 'r') as f:
        session_dict = json.load(f)
    if 'phase_shift' not in session_dict:
        phase_vs_position = initialize((PhaseVsPosition, ), session, experimental_group_name)['PhaseVsPosition']
        phase_vs_position.find_theta_0(min_shift=-60, max_shift=60, num_shifts=61)

# run analyses and plot results
for session in sessions:
    print(f"\nAnalyzing session {session}...")

    for group_name in group_names:
        analyze(session, group_name)




