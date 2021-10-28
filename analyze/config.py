import os
import sys
import json
import pickle
from socket import gethostname
import matplotlib.pyplot as plt


# CHANGE THIS ACCORDING TO YOUR FILE SYSTEM!
# paths to the data, the pickles and the figures
host_name = gethostname()
if host_name == "lux39":
    external_drive_path = "/media/eparra"
    internal_drive_path = "/local"
elif host_name == "etp":
    external_drive_path = "/media/eloy"
    internal_drive_path = "/c/DATA"
else:
    sys.exit("Host not recognized. Add host name and paths to the data and the pickles in 'config.py'.")

if os.path.exists(external_drive_path):
    drive_path = external_drive_path
else:
    drive_path = internal_drive_path

data_path = f"{drive_path}/INTENSO/RatData"  # Path to the folder containing the folders hc-3 and hc-11
pickles_path = f"{data_path}/pickles"  # Path where the program will save pickles
figures_path = f"{data_path}/figures"  # Path to where the program will save figures


# select parameters
parameters_id = "paper"
with open('parameters/' + parameters_id + '.json') as parameters_file:
    general_parameters = json.load(parameters_file)

speed_groups = general_parameters['ALL']['speed_groups']

experimental_group_name = "EXPERIMENTAL"

pickle_results = 1


# plotting parameters
save_figures = 1
figure_format = 'pdf'
small_plots = 1

plt.rcParams["savefig.format"] = figure_format
plt.rcParams["savefig.dpi"] = 300  # in case figures are saved as png

cm = 1 / 2.54

if small_plots:
    tick_size = 1.5
    # font_size = 6
    # line_width = 0.75

    font_size = 7
    line_width = 1

    thin_line_width = 0.75  # 0.5

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': 'Arial',
                         'font.size': font_size, 'mathtext.default': 'regular', 'axes.linewidth': thin_line_width,
                         'xtick.major.width': thin_line_width, 'xtick.major.size': tick_size,
                         'xtick.minor.size': tick_size*0.6, 'ytick.major.width': thin_line_width,
                         'ytick.major.size': tick_size, 'ytick.minor.size': tick_size*0.6,
                         'lines.linewidth': line_width, 'lines.markersize': 4, 'lines.markeredgewidth': 0.0,
                         'legend.columnspacing': 0.3, 'axes.labelpad': 3,
                         'legend.handletextpad': 0.2,
                         'xtick.major.pad': 2, 'ytick.major.pad': 2,
                         })

