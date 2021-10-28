import json
import matplotlib.pyplot as plt
from data_analysis.lfp import LFP
from data_analysis.analyze.config import data_path, general_parameters, experimental_group_name, save_figures, \
    figures_path


# session = "ec013.412"
# session = "ec014.468"
# session = "ec014.639"
# session = "ec016.233"
# session = "Buddy.06272013"
session = "Achilles.10252013"
# session = "Gatsby.08022013"

p = general_parameters['LFP']

with open('sessions/' + session + '.json') as session_file:
    s = json.load(session_file)

lfp = LFP(session, experimental_group_name, "LFP", filter_order=p['filter_order'],
          bandpass_frequencies=p['bandpass_frequencies'], save_figures=False, figures_path=figures_path)

lfp.load_lfp(data_path, s['dataset'], s['session_set'], s['session'], channels=s['lfp_channels'])
lfp.finish_initialization()

# lfp.phase_from_waveform()

lfp.phase_from_hilbert_transform()
lfp.find_cycle_boundaries()

# lfp.phase_from_peaks()

lfp.find_significant_theta(amplitude_percentile=p['significance_percentile'], plot_steps=False, plot_histogram=False)
lfp.find_significant_cycles()
lfp.calculate_phase_locking(data_path, s['dataset'], s['session_set'], s['session'], s['discarded_intervals'], plot=True)
lfp.comp_instantaneous_frequency()

# lfp.plot(time_interval=(0, 500), channels=[0])
# lfp.plot(time_interval=(17079.2, 17300), channels=[24])  # Gatsby
lfp.plot(time_interval=(18080, 182000), channels=[0])  # Achilles

plt.show()
