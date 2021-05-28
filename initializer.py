import os
import glob
import json
import xml.etree.ElementTree as ET
from data_analysis.spikes import SpikesBase, ExpSpikes, UniformSpikes, VariableSpikes, SpeedSpikes
from data_analysis.analyze.config import general_parameters, data_path, pickle_results, pickles_path, save_figures, \
    figure_format, figures_path


def initialize(classes_needed, super_group_name, group_name):
    """Initialize a set of classes inherited from my Base class.
    
    Args:
        classes_needed (tuple): Tuple of classes to initialize (only the ones needed in the end, intermediate classes
            required to construct them will be handled automatically).
        super_group_name (string): Name of the super-group (e.g., session name).
        group_name (string): Name of the group.

    Returns:
        dict: Dictionary containing initialized instances keyed by their class names.
    """

    def initialize_class(class_def):
        child_name = SpikesBase.__name__ if "Spikes" in class_def.__name__ else class_def.__name__

        if class_def in [UniformSpikes, VariableSpikes, SpeedSpikes]:
            parameters_name = f'{class_def.__name__}|{group_name}'
        else:
            parameters_name = class_def.__name__

        return class_def.default_pickle(super_group_name, group_name, child_name,
                                        {**session_parameters, **general_parameters[parameters_name]},
                                        save_figures, figure_format, figures_path, pickle_results, pickles_path,
                                        data_path=data_path)

    # load session parameters
    with open('sessions/' + super_group_name + '.json') as session_file:
        session_parameters = json.load(session_file)

    # recursively initialize the classes
    initialized = []
    instances = {}

    def recursive_initializer(class_def):
        if class_def is SpikesBase:
            if 'Variable' in group_name:
                class_def = VariableSpikes
            elif any([name in group_name for name in ['Time', 'Position']]):
                class_def = UniformSpikes
            elif 'SpeedDep' in group_name:
                class_def = SpeedSpikes
            else:
                class_def = ExpSpikes

        class_name = class_def.__name__
        if class_name not in initialized:

            # initialize dependencies if not done yet
            for dependency in class_def.dependencies:
                if dependency.__name__ not in initialized:
                    recursive_initializer(dependency)

            # initialize
            instance = initialize_class(class_def)
            if class_def in classes_needed:
                instances[class_name] = instance
            initialized.append(class_name)

    for class_def in classes_needed:
        recursive_initializer(class_def)

    return instances


def initialize_session_files(dataset, experimental_group_name, lfp_channel_num=0, sessions_folder='sessions'):
    """Initialize files containing session-specific parameters for all sessions in a dataset.

    Args:
        dataset (string): Name of the dataset, e.g., hc-3.
        experimental_group_name (string): Name used for the experimental group.
        lfp_channel_num (int): LFP channel to choose for each electrode. Defaults to 0.
        sessions_folder (string): Path to the folder where session files are stored.
    """
    for folder in glob.glob(f'{data_path}/{dataset}/**/'):

        if not any(x in folder for x in ['metadata', 'additional', 'docs', 'kamran']):
            session_set = folder.split('/')[-2]
            print(f'\nInitializing session set: {session_set}...')

            for subfolder_num, subfolder in enumerate(glob.glob(f'{folder}/**/')):
                session = subfolder.split('/')[-2]
                print(session)
                file_path = f'{sessions_folder}/{session}.json'

                if not os.path.exists(file_path):
                    if subfolder_num == 0:
                        # find relevant lfp channels
                        xml_tree = ET.parse(f'{subfolder}{session}.xml')
                        lfp_groups = xml_tree.find('anatomicalDescription/channelGroups')

                        spikes = ExpSpikes(session, experimental_group_name, SpikesBase.__name__, data_path, dataset,
                                           session_set, session).spikes

                        lfp_channels = []
                        for electrode_num, lfp_group in enumerate(lfp_groups):
                            if electrode_num + 1 in spikes.electrodes:
                                lfp_channels.append(int(lfp_group[lfp_channel_num].text))

                        if len(spikes.electrodes) == 0:
                            print(f'Session set {session_set} does not have any valid cells!')
                        else:
                            print(f'relevant lfp channels: {lfp_channels}')

                    # create dictionary of session parameters and save it
                    session_parameters = {'dataset': dataset, 'session_set': session_set, 'session': session,
                                          'lfp_channels': lfp_channels}

                    with open(file_path, 'w') as session_file:
                        json.dump(session_parameters, session_file, indent=2)
