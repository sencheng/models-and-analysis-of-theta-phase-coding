import glob
import pandas
import h5py
import numpy as np
from scipy.io import loadmat
import xml.etree.ElementTree as ET
from data_analysis.general import Base


class Spikes:
    """Container class for spiking data.

    Attributes:
        electrodes (list(int)): List of electrodes.
        electrode_cluster_pairs (list(tuple(int))): List of (electrode, cluster_id) pairs defining each identified cell.
        spike_times (list(list(list(float)))): The spike times for each cluster id for each electrode (s).
    """
    def __init__(self):
        self.electrodes = []
        self.electrode_cluster_pairs = []
        self.spike_times = []


class SpikesBase(Base):
    """Base class for spikes.

    Args:
        super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is defined
            as belonging to the super-group, it will be shared across sub-groups.
        group_name (string): Name of the low-level sub-group used for pickles and figures.
        child_name (string): Name of the instance used for pickles and figures.

    Attributes:
        spikes (SpikesContainer): Container of spiking data.
    """
    def __init__(self, super_group_name, group_name, child_name, save_figures=False, figure_format="png",
                 figures_path="figures"):

        super().__init__(super_group_name, group_name, child_name, save_figures, figure_format, figures_path)

        self.spikes = Spikes()


def load_spikes(data_path, dataset, session_set, session, discarded_intervals, region, cell_type='p',
                diba_clusters=False):
    spikes = Spikes()
    path = f'{data_path}/{dataset}/{session_set}/{session}/{session}'

    if dataset == 'hc-11':
        with h5py.File(f'{path}_sessInfo.mat', 'r') as f:
            spike_times = np.array(f['sessInfo']['Spikes']['SpikeTimes'])[0]
            maze_epoch = np.array(f['sessInfo']['Epochs']['MazeEpoch'])
            first_index = np.searchsorted(spike_times, maze_epoch[0])[0]
            last_index = np.searchsorted(spike_times, maze_epoch[1], side='right')[0]
            spike_times = spike_times[first_index:last_index]
            spike_ids = np.array(f['sessInfo']['Spikes']['SpikeIDs'])[0, first_index:last_index].astype(int)
            pyr_ids = np.array(f['sessInfo']['Spikes']['PyrIDs'])[0].astype(int)

            for discarded_interval in discarded_intervals:
                index_start = np.searchsorted(spike_times, discarded_interval[0])
                index_end = np.searchsorted(spike_times, discarded_interval[1], side='right')
                spike_times = np.delete(spike_times, slice(index_start, index_end))
                spike_ids = np.delete(spike_ids, slice(index_start, index_end))

            spikes.electrodes = np.unique((pyr_ids / 100).astype(int)).tolist()
            for pyr_id in np.unique(pyr_ids):
                spikes.electrode_cluster_pairs.append([int(pyr_id / 100), int(pyr_id % 100)])
                spikes.spike_times.append(spike_times[spike_ids == pyr_id])
    else:
        # load metadata table
        metadata_table_path = glob.glob(f'{data_path}/{dataset}/*/*tables.xlsx')[0]
        excel_metadata_table = pandas.read_excel(metadata_table_path)
        first_column, last_column, headers_row = 1, 15, 4  # location of the metadata block within the excel file
        metadata_table = excel_metadata_table.iloc[headers_row - 1:, first_column - 1:last_column]
        metadata_table.columns = excel_metadata_table.iloc[headers_row - 2, first_column - 1:last_column]

        # select rows that fulfill the conditions, and read electrodes and cluster ids
        selected_rows = metadata_table[
            metadata_table['topdir'].eq(session_set) & metadata_table['region'].eq(region) & metadata_table[
                'type'].eq(cell_type)]

        spikes.electrodes = sorted(set(selected_rows['ele']))
        cluster_ids = []

        if session.split('.')[0] in ['gor01', 'vvp01'] and diba_clusters:

            mat_file_paths = glob.glob(f'{data_path}/{dataset}/diba_mats/*.mat')
            for mat_file_path in mat_file_paths:
                IIdata = loadmat(mat_file_path)['IIdata']
                for session_data in IIdata:
                    if session_data['name'][0][0] == session:
                        for electrode in spikes.electrodes:
                            cluster_ids.append([i + 1 for i, cluster in
                                                enumerate(session_data['cluq2'][0][0][electrode - 1][0])
                                                if cluster not in [0, 3, 5]])

            clu_files_path = f'{data_path}/{dataset}/{session_set}/{session}/diba/{session}'

        else:
            for electrode in spikes.electrodes:
                cluster_ids.append([int(value) for value in
                                    selected_rows[selected_rows['ele'].eq(electrode)]['clu'].values])
            clu_files_path = path

        # get sampling rate from xml file
        xml_tree = ET.parse(f'{path}.xml')
        sampling_rate = int(xml_tree.find('acquisitionSystem/samplingRate').text)

        # load spikes
        for electrode, electrode_cluster_ids in zip(spikes.electrodes, cluster_ids):
            spike_times_file_path = f'{clu_files_path}.res.{electrode}'
            unclassified_spike_times = np.loadtxt(spike_times_file_path, dtype=int) / sampling_rate
            spike_cluster_ids = np.loadtxt(f'{clu_files_path}.clu.{electrode}', dtype=int)[1:]

            for discarded_interval in discarded_intervals:
                index_start = np.searchsorted(unclassified_spike_times, discarded_interval[0])
                index_end = np.searchsorted(unclassified_spike_times, discarded_interval[1], side='right')
                unclassified_spike_times = np.delete(unclassified_spike_times, slice(index_start, index_end))
                spike_cluster_ids = np.delete(spike_cluster_ids, slice(index_start, index_end))

            for cluster_id in electrode_cluster_ids:
                spike_times = unclassified_spike_times[spike_cluster_ids == cluster_id]
                if len(spike_times) != 0:
                    spikes.spike_times.append(spike_times)
                    spikes.electrode_cluster_pairs.append([electrode, cluster_id])
    return spikes
