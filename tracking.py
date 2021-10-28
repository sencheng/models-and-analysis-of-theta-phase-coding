import sys
import h5py
import numpy as np
from scipy.ndimage import filters
from scipy.stats import linregress
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from data_analysis.general import Base, interpolate_missing
from data_analysis.lfp import LFP


class LED:
    """LED tracking data.

    Args:
        x (np.array): Time series containing the x coordinates of the LED obtained from the tracking system (cm).
        y (np.array): Time series containing the y coordinates of the LED obtained from the tracking system (cm).
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y


class Tracking(Base):
    """Class for loading and working with position data from the datasets published by Mizuseki et al., 2014.
    The folder structure and file names of the dataset must be left unaltered for this class to work properly.

    Args:
        super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is defined
            as belonging to the super-group, it will be shared across sub-groups.
        group_name (string): Name of the low-level sub-group used for pickles and figures.
        child_name (string): Name of the instance used for pickles and figures.
        corner_size (float): Size of the corners (cm) for calculating the characteristic speed.
        min_central_speed (float): Minimum speed outside of the corner (cm/s) for calculating the characteristic speed.
        save_figures (bool): Whether to save the figures.

    Attributes:
        sampling_rate (float): Sampling rate of the tracking system.
        times (np.array): Time points (s).
        LEDs (dict): Dictionary containing 'front' and 'back' LED instances.
        x (np.array): X coordinates interpolated between the two LEDs (cm).
        y (np.array): Y coordinates interpolated between the two LEDs (cm).
        speed_2D (np.array): Smoothed derivative of position (cm/s).
        d (np.array): Displacement calculated by projecting x and y coordinates onto the fitted line (cm).
        d_interpolated (np.array): Displacement with missing values filled by interpolation (cm).
        interpolation_range (slice): Slice of time indices excluding time points with invalid values at the beginning
            and end of the session.
        d_runs (np.array): d subtracting the minimum valid position associated to a traversal (cm).
        d_runs_offset (float): Minimum valid position associated to a traversal (cm).
        d_runs_span (float): Max(d_runs) - min(d_runs) (cm).
        speed_1D (np.array): Absolute value of the smoothed derivative of the displacement.
        run_type (np.array): Run type (-1: no type, 0: forward run, 1: backward run).
        num_runs (list(int)): Number of runs in the forward and backward runs.
        run_num (np.array): For each time point, number of the run.
        fit_slope (float): Slope of the linear fit to the position data (cm/s).
        fit_intercept (float): Intercept of the linear fit to the position data (cm).
    """

    belongs_to_super_group = True
    dependencies = (LFP,)

    def __init__(self, super_group_name, group_name, child_name, spatial_bin_size=5, corner_size=40,
                 min_central_speed=10, save_figures=False, figure_format="png", figures_path="figures",
                 pickle_results=True, pickles_path="pickles"):

        super().__init__(super_group_name, group_name, child_name, save_figures, figure_format, figures_path,
                         pickle_results, pickles_path)

        self.spatial_bin_size = spatial_bin_size
        self.num_spatial_bins = None
        self.corner_size = corner_size
        self.min_central_speed = min_central_speed

        self.x = None
        self.y = None
        self.times = None
        self.sampling_rate = None
        self.significant_theta = None
        self.LEDs = None
        self.speed_2D = None
        self.d = None
        self.d_interpolated = None
        self.interpolation_range = None
        self.d_runs_offset = None
        self.d_runs = None
        self.d_runs_span = 0
        self.corners = []
        self.speed_1D = None
        self.acceleration_1D = None
        self.run_type = None
        self.run_type_names = ["Rightward run", "Leftward run"]
        self.num_runs = None
        self.run_num = None
        self.fit_slope = 0
        self.fit_intercept = 0
        self.top_speeds = None
        self.characteristic_speeds = None
        self.min_speeds = None
        self.mean_acceleration = None

    def load_tracking(self, data_path, dataset, session_set, session, lfp, discarded_intervals=(),
                      back_to_front_progress=None, sampling_rate=39.0625):
        """Load the tracking data.

        Args:
            data_path (string): Path to the data.
            dataset (string): Name of the dataset, e.g., hc-3.
            session_set (string): Name of the set of sessions that get lumped together, e.g., ec013.28.
            session (string): Name of the session, e.g., ec013.412.
            discarded_intervals (tuple(tuple(float)): Intervals to be discarded as a tuple of (start, stop) pairs (s).
            back_to_front_progress (float): Progress between the back LED (0) up to the front LED (1). Can also take
                values below 0 and above 1 to take points beyond either LED along the line uniting them.
            sampling_rate (float): Sampling rate of the tracking system (Hz).
        """

        def discard_intervals():
            for discarded_interval in discarded_intervals:
                positions[(discarded_interval[0] <= self.times) & (self.times < discarded_interval[1])] = np.nan

        path = f'{data_path}/{dataset}/{session_set}/{session}/{session}'

        if dataset == 'hc-11':
            with h5py.File(f'{path}_sessInfo.mat', 'r') as f:
                self.times = np.squeeze(np.array(f['sessInfo']['Position']['TimeStamps']))
                sampling_times, counts = np.unique(np.diff(self.times), return_counts=True)
                self.sampling_rate = 1 / (np.dot(sampling_times, counts) / np.sum(counts))
                positions = np.array(f['sessInfo']['Position']['TwoDLocation']).T * 100
                discard_intervals()
                self.x = positions[:, 0]
                self.y = positions[:, 1]
        else:
            positions = np.loadtxt(f'{path}.whl')
            positions[positions < 0] = np.nan

            self.sampling_rate = sampling_rate
            self.times = np.arange(len(positions)) / self.sampling_rate

            discard_intervals()

            self.LEDs = {'front': LED(x=positions[:, 0], y=positions[:, 1]),
                         'back': LED(x=positions[:, 2], y=positions[:, 3])}

            self.rat_position_from_LEDs(back_to_front_progress)

        significant_theta = []
        for time in self.times:
            significant_theta.append(lfp.at_time(time, channel_index=0, return_significance=True)[0])
        self.significant_theta = np.array(significant_theta).astype(bool)

    def generate_step_trajectories(self, track_length, duration, inter_trial_duration, sections, speeds,
                                   sampling_rate=40):
        """Generate trajectories running at different constant speeds in different sections.
        """
        self.sampling_rate = sampling_rate
        self.times = np.arange(0, duration, 1/sampling_rate)
        self.significant_theta = np.ones(self.times.size)
        self.x = []
        self.y = np.zeros(self.times.size)

        def get_speed():
            for section_num, section in enumerate(sections):
                if section[0] <= x <= section[1]:
                    return speeds[section_num]
            return 0

        x = 0
        in_corner = True
        run_type = 1
        inter_trial_counter = 0
        inter_trial_steps = inter_trial_duration * sampling_rate
        for time_step in range(self.times.size):
            if in_corner:
                if inter_trial_counter >= inter_trial_steps:
                    in_corner = False
                    inter_trial_counter = 0
                    run_type = 1 - run_type
                else:
                    inter_trial_counter += 1
            else:
                x += (1 - 2*run_type) * get_speed() / sampling_rate
                if x <= 0:
                    x = 0
                    in_corner = True
                elif x >= track_length:
                    x = track_length
                    in_corner = True

            self.x.append(x)
        self.x = np.array(self.x)

    def rat_position_from_LEDs(self, back_to_front_progress):
        """Interpolate points between the two LED positions.

        Args:
            back_to_front_progress (float): Progress between the back LED (0) up to the front LED (1). Can also take
                values below 0 and above 1 to take points beyond either LED along the line uniting them.
        """
        self.x = (self.LEDs['front'].x - self.LEDs['back'].x) * back_to_front_progress + self.LEDs['back'].x
        self.y = (self.LEDs['front'].y - self.LEDs['back'].y) * back_to_front_progress + self.LEDs['back'].y

    def calculate_speed_2D(self, sigma, plot=False):
        """Calculate running speeds and smooth them with a Gaussian filter.

        Args:
            sigma (float): Standard deviation for the Gaussian filter (s).
            plot (bool): Plot raw and smoothed speeds.
        """
        raw_speed_2D = np.append((np.sqrt((np.diff(self.x)) ** 2 + (np.diff(self.y)) ** 2) * self.sampling_rate),
                                 np.nan)
        self.speed_2D = filters.gaussian_filter1d(raw_speed_2D, sigma * self.sampling_rate)

        if plot:
            fig, ax = plt.subplots()
            ax.plot(self.times, raw_speed_2D, label='raw speed')
            ax.plot(self.times, self.speed_2D, label='smoothed speed')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Running speed (cm/s)')
            ax.legend(loc='upper right')
            self.maybe_save_fig(fig, "2D_speed")

    def linear_fit(self, min_speed_ratio):
        """The linear fit is calculated using only points corresponding to high running speeds in order to avoid
        biasing the fit by asymmetric behavior at the feeder sites.

        Args:
            min_speed_ratio (float): Minimum ratio to the 95th percentile for including a point in the linear fit
                calculation (cm/s).
        """
        valid = ~np.isnan(self.speed_2D)
        valid[valid] &= (self.speed_2D[valid] > min_speed_ratio*np.percentile(self.speed_2D[valid], 95))
        self.fit_slope, self.fit_intercept = linregress(self.x[valid], self.y[valid])[:2]

    def project(self):
        """Project interpolated points onto a best fitting line.
        """
        unit_vector = np.array((1, self.fit_slope)) / np.sqrt(1 + self.fit_slope ** 2)
        self.d = np.empty(len(self.x))
        for pos_num, (x, y) in enumerate(zip(self.x, self.y)):
            self.d[pos_num] = np.dot(unit_vector, np.array((x, y)))
        self.d -= np.nanmin(self.d)

        # interpolate missing points
        self.d_interpolated, self.interpolation_range = interpolate_missing(self.d)

    def split_full_runs(self, in_corner_sigma=0.2, out_of_corner_sigma=1, min_speed=2, corner_sizes=(30, 30),
                        plot_steps=False):
        """Classify each time point as belonging to a forward or backward run. The displacement signal is smoothed with
        a Gaussian filter and the sign of its 1st difference determines run category.

        Args:
            in_corner_sigma (float): Standard deviation for the Gaussian filter relevant for positions within the
                corners (s).
            out_of_corner_sigma (float): Standard deviation for the Gaussian filter relevant for positions outside
                of the corners (s).
            min_speed (float): Threshold value for the absolute value of the speed in order to qualify as a run (cm/s).
            corner_sizes (tuple(float)): Size of the left and right corners (cm).
                Runs must span the complete extent between corners in order to qualify.
            plot_steps (bool): Plot intermediate results, including a version of the displacement signal with nans
                substituted by interpolation, and the smoothed signal.
        """
        upper_corner = np.nanmax(self.d) - corner_sizes[1]
        self.corners = (corner_sizes[0], upper_corner)

        # determine run type based on smoothed speed
        d_smooth_fine = filters.gaussian_filter1d(self.d_interpolated, in_corner_sigma * self.sampling_rate,
                                                  mode='nearest')
        d_smooth_coarse = filters.gaussian_filter1d(self.d_interpolated, out_of_corner_sigma * self.sampling_rate,
                                                    mode='nearest')
        d_diff_fine = np.append(np.diff(d_smooth_fine), 0)
        d_diff_coarse = np.append(np.diff(d_smooth_coarse), 0)
        min_diff = min_speed/self.sampling_rate
        self.run_type = np.full(len(self.d), -1)
        in_corner = np.full(self.interpolation_range.stop - self.interpolation_range.start, False)
        not_nan = ~np.isnan(self.d[self.interpolation_range])
        in_corner[not_nan] = ((self.d[self.interpolation_range][not_nan] <= self.corners[0]) |
                              (self.d[self.interpolation_range][not_nan] >= self.corners[1]))
        self.run_type[self.interpolation_range][d_diff_coarse > min_diff] = 0
        self.run_type[self.interpolation_range][d_diff_coarse < -min_diff] = 1
        self.run_type[self.interpolation_range][in_corner & (d_diff_fine > min_diff)] = 0
        self.run_type[self.interpolation_range][in_corner & (d_diff_fine < -min_diff)] = 1

        if plot_steps:
            fig, ax = plt.subplots()
            ax.plot(self.times[self.interpolation_range], self.d_interpolated, 'C1', label='interpolated')
            ax.plot(self.times, self.d, 'C7', label='original')
            ax.plot(self.times[self.interpolation_range], d_smooth_fine, color='C0', label='smoothed, fine')
            ax.plot(self.times[self.interpolation_range], d_smooth_coarse, color='C6', label='smoothed, coarse')
            ax.axhline(corner_sizes[0], color='k', linestyle='dotted')
            ax.axhline(upper_corner, color='k', linestyle='dotted')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Displacement (cm)')
            ax.legend(loc='upper right')
            self.maybe_save_fig(fig, "splitting_steps")

        # connect isolated stretches
        def operate_on_islands(island_value, operation):
            start = 0
            while island_value in self.run_type[start:]:
                beginning = start + np.argmax(self.run_type[start:] == island_value)
                end = beginning + np.argmax(self.run_type[beginning:] != island_value)
                if end == beginning:  # run_type ends in a stretch of island values
                    break
                else:
                    operation(beginning, end)
                start = end

        def fill_in(beginning, end):
            if self.d[beginning] >= corner_sizes[0] and self.d[end] <= upper_corner:
                if beginning > 0 and self.run_type[beginning - 1] == self.run_type[end]:
                    self.run_type[beginning:end] = self.run_type[end]

        operate_on_islands(island_value=-1, operation=fill_in)

        # remove incomplete traversals
        def remove_incomplete_traversal(beginning, end):
            if np.isnan(self.d[beginning:end]).all():
                self.run_type[beginning:end] = -1
            elif (np.nanmin(self.d[beginning:end]) > corner_sizes[0]
                  or np.nanmax(self.d[beginning-1:end]) < upper_corner):
                self.run_type[beginning:end] = -1

        for run_type in range(2):
            operate_on_islands(island_value=run_type, operation=remove_incomplete_traversal)

        # fill in related variables, set run_type of nans as -1
        self.count_runs()
        self.define_d_runs()
        self.num_spatial_bins = int(round(self.d_runs_span/self.spatial_bin_size)) + 1
        self.run_type[np.isnan(self.d)] = -1

    def count_runs(self):
        self.num_runs = [0, 0]
        self.run_num = np.full((2, len(self.d)), np.nan)
        type_changes = np.nonzero(self.run_type[1:] - self.run_type[:-1])[0] + 1
        for type_change, next_type_change in zip(type_changes[:-1], type_changes[1:]):
            run_type = self.run_type[type_change]
            if run_type != -1:
                self.run_num[run_type, type_change:next_type_change] = self.num_runs[run_type]
                self.num_runs[run_type] += 1

    def define_d_runs(self):
        self.d_runs_offset = np.nanmin(self.d[self.run_type != -1])
        self.d_runs = self.d - self.d_runs_offset
        self.d_runs_span = np.nanmax(self.d_runs[self.run_type != -1])

    def calculate_speed_1D(self, sigma, plot=False):
        """Calculates the absolute value of the 1D projection of running speed and smooths it with a Gaussian filter.

        Args:
            sigma (float): Standard deviation for the filter.
            plot (bool): Plot raw and smoothed velocities.
        """
        raw_speed_1D = np.diff(self.d_interpolated) * self.sampling_rate
        self.speed_1D = np.full(len(self.times), np.nan)
        speed_range = slice(self.interpolation_range.start, self.interpolation_range.stop - 1)
        # because of the difference, the speed corresponds to points shifted by half a time bin
        self.speed_1D[speed_range] = filters.gaussian_filter1d(raw_speed_1D, sigma * self.sampling_rate)
        self.run_type[self.interpolation_range.stop - 2] = -1
        # the acceleration is properly centered
        self.acceleration_1D = np.append(np.nan, np.diff(self.speed_1D)) * self.sampling_rate
        if plot:
            fig, ax = plt.subplots(2, sharex='col')
            ax[0].plot(self.times[speed_range], raw_speed_1D, label='raw velocity')
            ax[0].plot(self.times, self.speed_1D, label='smoothed velocity')
            ax[0].set_ylabel('1D projection of running velocity (cm/s)')
            ax[0].legend(loc='upper right')
            ax[1].plot(self.times, self.acceleration_1D)
            ax[1].set_ylabel('Acceleration (cm/s^2)')
            ax[1].set_xlabel('Time (s)')

            self.maybe_save_fig(fig, "1D_speed")

    def calculate_characteristic_speeds(self, top_percentile=95, bottom_speed_from="box", averaging_window_size=30,
                                        bottom_percentile=40, median=True, min_speed_count_percentile=1):
        """Calculates the mean speed for each spatial bin (and running direction) within some range of speed percentiles
        for that bin.

        Args:
            top_percentile (int): Percentile of speeds within a bin that sets the upper bound for calculating the mean.
            bottom_speed_from (string): How to define the lower bound for calculating the mean:
                * 'box': Zero in the corner and then min_speed.
                * 'window': Top percentile minus averaging_window_size
                * 'percentile': Bottom percentile.
            averaging_window_size (float): Size of the window of speeds below the top percentile (cm/s).
            bottom_percentile (int): Percentile of speeds within a bin that sets the lower bound
                for calculating the mean.
            median (bool): Calculates the median, otherwise, calculates the mean.
            min_speed_count_percentile (int): Minimum percentile of speed counts for a bin to be included.
        """
        speeds = [[[] for _ in range(self.num_spatial_bins)] for _ in range(2)]
        speed_counts = np.zeros((2, self.num_spatial_bins))
        for run_type in range(2):
            speed_1D = self.speed_1D[self.run_type == run_type]
            if run_type == 1:
                speed_1D *= -1
            for position, speed in zip(self.d_runs[self.run_type == run_type], speed_1D):
                if not np.isnan(speed):
                    spatial_bin_num = int(round(position/self.spatial_bin_size))
                    speeds[run_type][spatial_bin_num].append(speed)
                    speed_counts[run_type][spatial_bin_num] += 1

        # min_speed_count = int(np.percentile(np.array(speed_counts), min_speed_count_percentile))
        min_speed_count = 1

        self.top_speeds = np.full((2, self.num_spatial_bins), np.nan)
        self.characteristic_speeds = np.full((2, self.num_spatial_bins), np.nan)
        self.min_speeds = np.full((2, self.num_spatial_bins), np.nan)

        for run_type in range(2):
            for spatial_bin_num, bin_speeds in enumerate(speeds[run_type]):
                if len(bin_speeds) >= min_speed_count:
                    bin_speeds = np.array(bin_speeds)
                    self.top_speeds[run_type, spatial_bin_num] = np.percentile(bin_speeds, top_percentile)
                    if bottom_speed_from == 'percentile':
                        self.min_speeds[run_type, spatial_bin_num] = np.percentile(bin_speeds, bottom_percentile)
                    elif bottom_speed_from == 'window':
                        self.min_speeds[run_type, spatial_bin_num] = (self.top_speeds[run_type, spatial_bin_num]
                                                                      - averaging_window_size)
                    elif bottom_speed_from == 'box':
                        self.min_speeds[run_type] = 0
                        corner_bins = int(round(self.corner_size/self.spatial_bin_size))
                        self.min_speeds[run_type, corner_bins:self.num_spatial_bins-corner_bins] = self.min_central_speed
                    else:
                        sys.exit("Parameter 'bottom_speed_from' must be 'percentile', 'window' or 'box'.")

                    f = np.median if median else np.nanmean
                    self.characteristic_speeds[run_type, spatial_bin_num] = \
                        f(bin_speeds[(bin_speeds > self.min_speeds[run_type, spatial_bin_num]) &
                                     (bin_speeds <= self.top_speeds[run_type, spatial_bin_num])])

        self.maybe_pickle_results([self.characteristic_speeds[0], self.characteristic_speeds[1][::-1]],
                                  "characteristic_speeds")

    def characteristic_speed_at_position(self, position, run_type):
        previous_bin = int(position / self.spatial_bin_size)
        next_bin = min(previous_bin + 1, self.num_spatial_bins - 1)
        remainder = position / self.spatial_bin_size - previous_bin
        return (self.characteristic_speeds[run_type, previous_bin] * (1 - remainder)
                + remainder * self.characteristic_speeds[run_type, next_bin])

    def calculate_histograms(self, num_spatial_bins, spatial_bin_size, ys, num_y_bins, y_bin_size,
                             threshold_y=0., ):
        y_vs_position = np.zeros((2, num_y_bins, num_spatial_bins))

        for run_type in range(2):
            ys_run_type = ys[self.run_type == run_type]
            if run_type == 1:
                ys_run_type *= -1
            for position, y in zip(self.d_runs[self.run_type == run_type], ys_run_type):
                if not np.isnan(position):
                    spatial_bin_num = int(round(position / spatial_bin_size))
                    if y >= threshold_y:
                        y_bin_num = int(round((y - threshold_y) / y_bin_size))
                        y_vs_position[run_type, y_bin_num, spatial_bin_num] += 1

        return y_vs_position

    @staticmethod
    def plot_histogram(axes, y_vs_position, min_d, num_spatial_bins, spatial_bin_size, threshold_y,
                       num_y_bins, y_bin_size, max_v, logarithm=True, plot_colorbar=False,
                       x_label="Position (cm)"):

        if logarithm:
            y_vs_position += 1
            norm = colors.LogNorm(vmin=1, vmax=max_v)
        else:
            norm = colors.Normalize(vmin=0, vmax=max_v)

        mat = axes.matshow(y_vs_position,
                           origin="lower", norm=norm,
                           extent=(min_d - spatial_bin_size / 2,
                                   min_d + (num_spatial_bins - 0.5) * spatial_bin_size,
                                   threshold_y - y_bin_size / 2,
                                   threshold_y + (num_y_bins - 0.5) * y_bin_size))
        axes.set_aspect("auto")
        if x_label is not None:
            axes.xaxis.set_ticks_position("bottom")
            axes.set_xlabel(x_label)
        axes.set_ylim(bottom=threshold_y - y_bin_size / 2)

        if plot_colorbar:
            bar = plt.colorbar(mat, ax=axes, aspect=40)
            bar.ax.set_ylabel(f"Count{' + 1' if logarithm else ''}")
        return mat

    def speed_vs_position(self, speed_bin_size, threshold_speed=0, plot_boundaries=True, logarithm=True,
                          fig_size=(8, 6)):
        """Plot heat maps with histograms of running speeds per spatial bin.

        Args:
            speed_bin_size (float): Size of the speed bin.
            threshold_speed (float): Minimum speed to consider.
            plot_boundaries (bool): Whether to plot the boundaries used in calculating the mean speed.
            logarithm (bool): Plot the logarithm of the bin count instead of the count.
            fig_size (tuple(float)): Size of the figure in inches.
        """
        max_speed = np.nanmax(np.abs(self.speed_1D[self.run_type != -1]))*1.1
        num_speed_bins = int(round((max_speed - threshold_speed) / speed_bin_size)) + 1

        speeds_vs_position = self.calculate_histograms(self.num_spatial_bins, self.spatial_bin_size, self.speed_1D,
                                                       num_speed_bins, speed_bin_size, threshold_speed)

        fig, ax = plt.subplots(1, 2, sharey="row", constrained_layout=True, figsize=fig_size)
        ax[0].set_ylabel("Running speed (cm/s)")

        positions = np.arange(self.num_spatial_bins) * self.spatial_bin_size + self.d_runs_offset

        max_v = np.max(speeds_vs_position)
        for run_type, run_type_name in enumerate(self.run_type_names):
            self.plot_histogram(ax[run_type], speeds_vs_position[run_type], self.d_runs_offset, self.num_spatial_bins,
                                self.spatial_bin_size, threshold_speed, num_speed_bins, speed_bin_size, max_v,
                                logarithm, plot_colorbar=run_type == 1)
            ax[run_type].set_title(f"{run_type_name} run")

            ax[run_type].plot(positions, self.characteristic_speeds[run_type], 'C3')
            if plot_boundaries:
                ax[run_type].plot(positions, self.min_speeds[run_type], 'C3', linestyle='dotted')
                ax[run_type].plot(positions, self.top_speeds[run_type], 'C3', linestyle='dotted')

        self.maybe_save_fig(fig, "speed_histogram")

    def speeds_sizes_sketch(self, speed_bin_size, run_type, ds, ls, sigmas, rectangle_y_pos, threshold=0.15,
                            theta_box_size=50, fig_size=(6.2/2.54, 5.75/2.54), line_colors=('C7', 'C1'),
                            height_ratios=(1, 0.35, 0.15), v_pad=0.4):
        fig, axes = plt.subplots(3, 2, sharex="col", figsize=fig_size,
                                 gridspec_kw={'height_ratios': height_ratios, 'width_ratios': (1, 0.03)})
        ax = axes[:, 0]
        axes[1, 1].set_axis_off()
        axes[2, 1].set_axis_off()

        ax[0].set_ylabel("Running speed (cm/s)")
        ax[1].set_ylabel("Theta\ntrajectories")
        for key in ['top', 'left', 'right']:
            ax[1].spines[key].set_visible(False)
        ax[1].set_yticks([])
        ax[2].set_yticks([])
        ax[2].set_ylabel("Firing\nrate")
        ax[2].set_xlabel("Distance run (cm)")
        for key in ['top', 'left', 'right']:
            ax[2].spines[key].set_visible(False)
        fig.align_ylabels()

        max_speed = np.nanmax(np.abs(self.speed_1D[self.run_type != -1])) * 1.1
        num_speed_bins = int(round(max_speed / speed_bin_size)) + 1
        speeds_vs_position = self.calculate_histograms(self.num_spatial_bins, self.spatial_bin_size, self.speed_1D,
                                                       num_speed_bins, speed_bin_size, 0)
        mat = self.plot_histogram(ax[0], speeds_vs_position[run_type], self.d_runs_offset, self.num_spatial_bins,
                                  self.spatial_bin_size, 0, num_speed_bins, speed_bin_size, np.max(speeds_vs_position),
                                  logarithm=True, plot_colorbar=False, x_label=None)
        bar = fig.colorbar(mat, cax=axes[0, 1])
        bar.ax.set_ylabel("Count + 1")
        positions = np.arange(self.num_spatial_bins) * self.spatial_bin_size + self.d_runs_offset
        x_min = positions[np.argmax(~np.isnan(self.characteristic_speeds[run_type]))] - self.spatial_bin_size/2
        x_max = (positions[self.num_spatial_bins - 1 - np.argmax(~np.isnan(self.characteristic_speeds[run_type][::-1]))]
                 + self.spatial_bin_size/2)
        ax[0].set_xlim((x_min, x_max))
        ax[0].plot(positions, self.characteristic_speeds[run_type], 'C3')
        ax[0].plot(positions, self.min_speeds[run_type], 'C3', linestyle='dotted')
        x = np.linspace(self.d_runs_offset, positions[-1], 200)
        ax[1].set_ylim((0, 1))
        for col_num, (d, l, sigma) in enumerate(zip(ds, ls, sigmas)):
            half_width = np.sqrt(- 2*sigma**2 * np.log(threshold))
            ax[0].add_patch(plt.Rectangle((d - half_width, rectangle_y_pos[col_num][0]), half_width*2,
                                          rectangle_y_pos[col_num][1] - rectangle_y_pos[col_num][0], fill=False,
                                          ec=line_colors[col_num], linestyle='dashed'))

            ax_ins = inset_axes(ax[1], width="100%", height="100%",
                                bbox_to_anchor=(d - theta_box_size/2, v_pad, theta_box_size, 1-v_pad),
                                bbox_transform=ax[1].transData, loc='lower left', borderpad=0)
            ax_ins.set_ylim((-0.5, 0.5))
            ax_ins.set_xlim((0, 1))
            ax_ins.set_yticks([])
            if col_num == 0:
                ax_ins.set_ylabel("Position", fontsize='small')
            ax_ins.set_xticks([])
            ax_ins.set_xlabel(r"$\theta$ cycle", fontsize='small')
            ax_ins.axhline(0, color='C7', linestyle='dashed')
            ax_ins.plot(np.array((0, 1)), np.array((-l/2, l/2)), color='k')
            for key in ['top', 'bottom', 'left', 'right']:
                ax_ins.spines[key].set_color(line_colors[col_num])

            ax[2].plot(x, np.exp(-(x - d)**2 / (2*sigma**2)), color=line_colors[col_num])

        fig.tight_layout(h_pad=0.2)
        self.maybe_save_fig(fig, "sketch")

    def acceleration_vs_position(self, acceleration_bin_size, plot=True, logarithm=True, fig_size=(8, 6)):
        min_acceleration = np.nanmin(self.acceleration_1D[self.run_type != -1])
        max_acceleration = np.nanmax(self.acceleration_1D[self.run_type != -1])
        acceleration_bound = max(-min_acceleration, max_acceleration)
        num_acceleration_bins = int(round(2*acceleration_bound / acceleration_bin_size)) + 1
        acceleration_bound = (num_acceleration_bins - 1)/2 * acceleration_bin_size

        acceleration_vs_position = self.calculate_histograms(self.num_spatial_bins, self.spatial_bin_size,
                                                             self.acceleration_1D, num_acceleration_bins,
                                                             acceleration_bin_size, -acceleration_bound)
        self.mean_acceleration = [np.full(self.num_spatial_bins, np.nan) for _ in range(2)]
        a = np.linspace(-acceleration_bound, acceleration_bound, num_acceleration_bins)
        for run_type in range(2):
            for spatial_bin_num in range(self.num_spatial_bins):
                if np.sum(acceleration_vs_position[run_type][:, spatial_bin_num]):
                    self.mean_acceleration[run_type][spatial_bin_num] = \
                        np.average(a, weights=acceleration_vs_position[run_type][:, spatial_bin_num])

        if plot:
            fig, ax = plt.subplots(1, 2, sharey="row", constrained_layout=True, figsize=fig_size)
            ax[0].set_ylabel(r"Acceleration $(cm/s^2)$")

            max_a = np.max(acceleration_vs_position)
            positions = np.arange(self.num_spatial_bins) * self.spatial_bin_size + self.d_runs_offset

            for run_type, run_type_name in enumerate(self.run_type_names):
                self.plot_histogram(ax[run_type], acceleration_vs_position[run_type], self.d_runs_offset,
                                    self.num_spatial_bins,
                                    self.spatial_bin_size, -acceleration_bound, num_acceleration_bins, acceleration_bin_size,
                                    max_a, logarithm, plot_colorbar=run_type == 1)
                ax[run_type].set_title(f"{run_type_name} run")
                ax[run_type].plot(positions, self.mean_acceleration[run_type], 'C3')
                ax[run_type].axhline(0, linestyle='dashed', color='white')

            self.maybe_save_fig(fig, "acceleration_histogram")

    def at_time(self, time, d_runs=False, return_run_num=False, return_speed=False, return_significant_theta=False):
        """Calculates the displacement at a certain time point by linear interpolation between the two nearest
        data points, as well as the the run type (forward, backward or none), run number and speed.

        Args:
            time (float): Time point (s).
            d_runs (bool): The minimum displacement with valid run type becomes the 0.
            return_run_num (bool): Return number the run number.
            return_speed (bool): Return the speed (positive when going in the corresponding running direction).
            return_significant_theta (bool): Return whether theta oscillation is significant.

        Returns:
            (tuple): A tuple containing:
                * d (float): The displacement (cm).
                * run_type (int): Run type for the time point (-1: no type, 0: forward run, 1: backward run)
                * run_num (int): Run number or nan.
                * speed (float): Running speed (cm/s).
        """

        next_time_step = min(len(self.times) - 1, np.searchsorted(self.times, time))
        previous_time_step = max(0, next_time_step - 1)
        remainder = (time - self.times[previous_time_step])*self.sampling_rate
        closest_time_step = previous_time_step if remainder < 0.5 else next_time_step

        returns = []

        if d_runs:
            returns.append(self.d_runs[previous_time_step] * (1 - remainder) + self.d_runs[next_time_step] * remainder)
        else:
            returns.append(self.d[previous_time_step] * (1 - remainder) + self.d[next_time_step] * remainder)

        if self.run_type[previous_time_step] == self.run_type[next_time_step]:
            run_type = self.run_type[previous_time_step]
        else:
            run_type = -1
        returns.append(run_type)

        if return_run_num:
            returns.append(self.run_num[run_type, closest_time_step].astype(int))

        if return_speed:
            if remainder <= 0.5:
                speed = ((0.5 - remainder) * self.speed_1D[max(0, previous_time_step - 1)] +
                         (0.5 + remainder) * self.speed_1D[previous_time_step])
            else:
                speed = (((1.5 - remainder) * self.speed_1D[previous_time_step]) +
                         ((remainder - 0.5) * self.speed_1D[next_time_step]))
            if run_type == 1:
                speed *= -1
            returns.append(speed)

        if return_significant_theta:
            returns.append(self.significant_theta[closest_time_step])

        return returns

    def plot_positions(self, start=0, end=None, plot_front_LED=True, plot_back_LED=True, plot_interpolated=True,
                       plot_fit=True):
        """Plot x, y coordinates of the animal within some time interval.

        Args:
            start (float): Start of the time interval (s).
            end (float): Optional: end of the time interval (s).
            plot_front_LED (bool): Plot position of the front LED.
            plot_back_LED (bool): Plot position of the back LED.
            plot_interpolated (bool): Plot position interpolated between the LEDs.
            plot_fit (bool): Plot linear fit.
        """
        if end is not None:
            end = int(round(end * self.sampling_rate))
        indices = slice(int(round(start * self.sampling_rate)), end)
        fig, ax = plt.subplots(figsize=(5, 3))
        colors = ['C7', 'C0', 'C1']
        if plot_front_LED:
            ax.plot(self.LEDs['front'].x[indices], self.LEDs['front'].y[indices], label='front LED')
        if plot_back_LED:
            ax.plot(self.LEDs['back'].x[indices], self.LEDs['back'].y[indices], label='back LED')

        if plot_interpolated:
            for i, run_type in enumerate((-1, 0, 1)):
                x = np.where(self.run_type[indices] == run_type, self.x[indices], np.nan)
                y = np.where(self.run_type[indices] == run_type, self.y[indices], np.nan)
                label = self.run_type_names[run_type] if run_type != -1 else ''
                ax.plot(x, y, label=label, color=colors[i])
        if plot_fit:
            x = (np.nanmin(self.x[indices]), np.nanmax(self.x[indices]))
            ax.plot(x, (self.fit_intercept + self.fit_slope * x[0], self.fit_intercept + self.fit_slope * x[1]),
                    color='k', label='fit')
        ax.axis('equal')
        ax.set_xlabel('x coordinate (cm)')
        ax.set_ylabel('y coordinate (cm)')
        ax.legend()
        fig.tight_layout()
        self.maybe_save_fig(fig, "positions")

    def plot_displacement(self, zoom=None, split_runs=True):
        """Plot displacement of the animal.

        Args:
            split_runs (bool): Overlay forward and backward runs in different colors.
            zoom (tuple(float)): Interval of time into which to zoom in (s).
        """
        fig, ax = plt.subplots()
        ax.plot(self.times, self.d, 'C7')
        if split_runs:
            d_forwards = np.full(len(self.d), np.nan)
            d_forwards[self.run_type == 0] = self.d[self.run_type == 0]
            d_backwards = np.full(len(self.d), np.nan)
            d_backwards[self.run_type == 1] = self.d[self.run_type == 1]
            ax.plot(self.times, d_forwards, 'C0')
            ax.plot(self.times, d_backwards, 'C1')

            axr = ax.twinx()
            for run_type in range(2):
                axr.plot(self.times, self.run_num[run_type], f'C{4+run_type}')
            axr.set_ylabel("Run number")

            ax.axhline(self.corners[0], color='k', linestyle='dotted')
            ax.axhline(self.corners[1], color='k', linestyle='dotted')

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Displacement (cm)')
        self.maybe_save_fig(fig, "displacements")

        if zoom:
            ax.set_xlim(zoom)
            self.maybe_save_fig(fig, "displacements_zoom")

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, p, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):
        tracking = cls(super_group_name, group_name, child_name, p['spatial_bin_size'], p['min_central_speed'],
                       p['spatial_bin_size'], save_figures, figure_format, figures_path, pickle_results,
                       pickles_path=pickles_path)
        tracking.load_tracking(kwargs['data_path'], p['dataset'], p['session_set'], p['session'], kwargs['LFP'],
                               p['discarded_intervals'], p['back_to_front_progress'], p['sampling_rate'], )
        tracking.calculate_speed_2D(p["speed_sigma"], plot=False)
        tracking.linear_fit(p['fitting_min_speed_ratio'])
        tracking.project()
        tracking.split_full_runs(p['runs_splitting_in_corner_sigma'], p['runs_splitting_out_of_corner_sigma'],
                                 p['runs_splitting_min_speed'], p['corner_sizes'], plot_steps=False)
        tracking.calculate_speed_1D(p["speed_sigma"], plot=False)
        tracking.calculate_characteristic_speeds(p['top_percentile'], p['bottom_speed_from'], median=p['median'])
        tracking.acceleration_vs_position(p['acceleration_bin_size'], plot=False)
        return tracking
