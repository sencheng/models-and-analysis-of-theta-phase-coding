import os
from data_analysis.analyze.config import experimental_group_name, figures_path


# select which sessions to analyze
sessions = [
    # "ec013.156",
    # "ec013.206",
    # "ec013.374",
    # "ec013.386",
    # "ec013.388",
    # "ec013.395",
    # "ec013.412",
    # "ec013.413",
    # "ec013.440",
    # "ec013.454",
    # "ec013.466",
    # "ec013.469",
    # "ec013.502",
    # "ec013.531",
    # only these three sessions were used in the models for ec013:
    "ec013.555",
    "ec013.556",
    "ec013.574",

    "ec014.468",
    "ec014.639",

    "ec016.233",
    "ec016.234",
    "ec016.269",

    # These discard the first 5 minutes of the recordings:
    "Achilles.10252013_alt",
    "Buddy.06272013_alt",
    "Cicero.09012014_alt",
    "Cicero.09172014_alt",

    # Original:
    # "Achilles.10252013",
    # "Buddy.06272013",
    # "Cicero.09012014",
    # "Cicero.09172014",
    # "Gatsby.08022013"  # No phase precession

    # "vvp01.1021",
    # "vvp01.1815",
    # "gor01.1315"
]

rats = []
for session in sessions:
    rat = session.split('.')[0]
    if rat not in rats:
        rats.append(rat)

# select what to analyze: experimental data and/or models
group_names = [
    # experimental_group_name,  # experimental data
    # 'Time',  # temporal sweep model
    # 'Position',  # spatial sweep model
    # 'SpeedDepVanilla'  # behavior-dependent sweep model

    # 'VariableNoise'  # variable noise model with extra variance
    'VariableNoiseVanilla'  #

    # 'SpeedDep57',  # behavior-dependent sweep model with extra variance
]

figures_path = f"{figures_path}/ALL"
for group_name in group_names:
    path = f"{figures_path}/{group_name}"
    if not os.path.isdir(path):
        os.makedirs(path)


