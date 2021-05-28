from data_analysis.spikes import ExpSpikes
from data_analysis.analyze.config import experimental_group_name
from data_analysis.initializer import initialize


session = "ec013.412"
# session = "Buddy.06272013"
# session = "gor01.1215"

spikes = initialize((ExpSpikes,), session, experimental_group_name)['ExpSpikes'].spikes

print('(electrode, cluster_id) pairs:\n', spikes.electrode_cluster_pairs)
print(spikes.electrodes)

# print some spike times as an example
electrode = 5
cluster = 3
pair_num = spikes.electrode_cluster_pairs.index([electrode, cluster])
print(f'spike times for electrode {electrode}, cluster {cluster}:\n{spikes.spike_times[pair_num]}')
