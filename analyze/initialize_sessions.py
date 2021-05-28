from data_analysis.initializer import initialize_session_files
from data_analysis.analyze.config import experimental_group_name


# Creates the session files from the metadata
initialize_session_files('hc-11', experimental_group_name)