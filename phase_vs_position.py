import os
import sys
import math
import copy
import json
import pickle
import numpy as np
from scipy import optimize
from scipy import odr
from scipy.stats import linregress
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
from data_analysis.general import Base
from data_analysis.lfp import LFP
from data_analysis.firing_fields import FiringFields


class PhaseVsPosition(Base):
    """Class for analyzing phase vs. position  relationships (phase precession).

        Args:
            super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is
                defined as belonging to the super-group, it will be shared across sub-groups.
            group_name (string): Name of the low-level sub-group used for pickles and figures.
            child_name (string): Name of the instance used for pickles and figures.
            lfp (LFP): LFP instance. Channels must be limited to include one channel per electrode.
            firing_fields (FiringFields): FiringFields instance.
            normalized_slope_bounds (tuple(float)): Normalized lower and upper bounds for the phase precession slope.
            orthogonal_fit_params (dict): Dictionary containing parameters for the orthogonal fit.
            save_figures (bool): Whether to save the figures.

        Attributes:
            num_unwrapping_iterations (int): In the orthogonal fit: number of unwrapping iterations.
            mean_error_threshold_factor (float): In the orthogonal fit: Factor multiplying the mean square error for
                determining the threshold that qualifies points to be moved.
            min_error_threshold (float): Minimum value for the square error threshold.
            orthogonal_init (bool): Save an orthogonal fit as the best fit before attempting to unwrap.
            circular_linear_step (bool): Use a circular_linear fit in the first unwrapping attempt.
            bottom_corner (tuple(float)): (x, y) coordinates of the upper right corner of the box starting in (0, 0)
                where points can be shifted by 360°.
            top_corner (tuple(float)): (x, y) coordinates of the lower left corner of the box ending in (1, 1)
                where points can be shifted by -360°.

            spike_times list((list(list(list(list(float)))))): List of spike times for each run through the environment,
                for each field, for each run type, for each (electrode, cluster id) pair.
            positions list((list(list(list(list(float)))))): Positions, same format as spike_times.
            phases list((list(list(list(list(float)))))): Phases, same format as spike_times.
            spike_speeds list((list(list(list(list(float)))))): Instantaneous running speeds, same format as spike_times.
            pass_spike_counts (list(list(list(list(float))))): Spike count for each run through the environment,
                for each field, for each run type, for each (electrode, cluster id) pair.
            pass_duration (list(list(list(list(float))))): Time duration (s) of each pass through the field,
                same format as pass_spike_counts.
            pass_speeds (list(list(list(list(float))))): Average speed for each pass through the field,
                same format as pass_spike_counts.
        """

    dependencies = (LFP, FiringFields)

    def __init__(self, super_group_name, group_name, child_name, lfp, firing_fields, only_significant_pp, num_shuffles,
                 percentile, normalized_slope_bounds, pass_min_speed, pass_speeds_from='tracking',
                 orthogonal_fit_params=None, save_figures=False, figure_format="png", figures_path="figures",
                 pickle_results=True, pickles_path="pickles"):

        super().__init__(super_group_name, group_name, child_name, save_figures, figure_format, figures_path,
                         pickle_results, pickles_path)

        self.firing_fields = firing_fields
        spikes = self.firing_fields.spikes
        self.tracking = self.firing_fields.tracking
        self.only_significant_pp = only_significant_pp

        self.normalized_slope_bounds = normalized_slope_bounds
        self.pass_min_speed = pass_min_speed

        if orthogonal_fit_params is not None:
            self.num_unwrapping_iterations = orthogonal_fit_params['num_unwrapping_iterations']
            self.mean_error_threshold_factor = orthogonal_fit_params['mean_error_threshold_factor']
            self.min_error_threshold = orthogonal_fit_params['min_error_threshold']
            self.orthogonal_init = orthogonal_fit_params['orthogonal_init']
            self.circular_linear_step = orthogonal_fit_params['circular_linear_step']
            self.bottom_corner = orthogonal_fit_params['bottom_corner']
            self.top_corner = orthogonal_fit_params['top_corner']

        pp_ok_folder = "fields_pp_ok"
        self.pp_ok_fields_path = f"{pp_ok_folder}/{self.super_group_name}.{self.group_name}.pkl"

        if only_significant_pp and os.path.exists(self.pp_ok_fields_path):
            with open(self.pp_ok_fields_path, 'rb') as fields_file:
                fields = pickle.load(fields_file)
            self.fields = {}
            for key in fields:
                self.fields[key] = [fields[key][i] for i, pp_ok in enumerate(fields['pp_ok']) if pp_ok]
        else:
            self.fields = firing_fields.screened_fields(include_incomplete=False)

        pair_trials = []
        for run_type in self.fields['run_types']:
            pair_trials.append([[] for _ in range(self.tracking.num_runs[run_type])])

        spike_times = copy.deepcopy(pair_trials)
        self.positions = copy.deepcopy(pair_trials)
        self.phases = copy.deepcopy(pair_trials)
        self.spike_speeds = copy.deepcopy(pair_trials)

        # go through each spike and fill containers
        for pair_num, (electrode_cluster_pair, pair_spike_times) in \
                enumerate(zip(spikes.electrode_cluster_pairs, spikes.spike_times)):
            if pair_num in self.fields['pair_nums']:
                electrode_index = spikes.electrodes.index(electrode_cluster_pair[0])

                for spike_time in pair_spike_times:
                    position, run_type, run_num, speed = self.tracking.at_time(spike_time, return_run_num=True,
                                                                               return_speed=True)
                    if run_type != -1:
                        phase = lfp.at_time(spike_time, electrode_index, return_phase=True)[0]
                        ok_fields = np.nonzero(((np.array(self.fields['pair_nums']) == pair_num) &
                                                (np.array(self.fields['run_types']) == run_type)))[0]
                        for field_num in ok_fields:
                            field_bounds = self.fields['bounds'][field_num]
                            if field_bounds[0] < position < field_bounds[1]:
                                spike_times[field_num][run_num].append(spike_time)
                                self.positions[field_num][run_num].append(position)
                                self.phases[field_num][run_num].append(phase)
                                self.spike_speeds[field_num][run_num].append(speed)
                                break

        # for each pass through a place field, calculate pass speed, spike count and duration
        self.pass_spike_counts = copy.deepcopy(pair_trials)
        self.pass_duration = copy.deepcopy(pair_trials)
        self.pass_speeds = copy.deepcopy(pair_trials)
        self.pass_speeds_variations = copy.deepcopy(pair_trials)
        self.pass_spreads = copy.deepcopy(pair_trials)

        for field_num, (run_type, field_bounds) in enumerate(zip(self.fields['run_types'], self.fields['bounds'])):
            for run_num in range(self.tracking.num_runs[run_type]):
                field_spike_times = np.array(spike_times[field_num][run_num])
                spike_speeds = np.array(self.spike_speeds[field_num][run_num])
                ok_spike_indices = np.array(spike_speeds) >= pass_min_speed
                self.pass_spike_counts[field_num][run_num] = np.sum(ok_spike_indices)
                if self.pass_spike_counts[field_num][run_num] > 1:
                    pass_duration = (max(field_spike_times[ok_spike_indices]) -
                                     min(field_spike_times[ok_spike_indices]))
                else:
                    pass_duration = 0

                d_ok = (((self.tracking.run_type == run_type) | (self.tracking.run_type == -1))
                        & (self.tracking.run_num[run_type] == run_num))
                d = self.tracking.d[d_ok]
                with np.errstate(invalid='ignore'):
                    if run_type == 0:
                        start_index = np.argmax(d >= field_bounds[0])
                        if np.max(d) > field_bounds[1]:
                            end_index = np.argmax(d > field_bounds[1])
                        else:
                            end_index = len(d)
                    else:
                        start_index = np.argmax(d <= field_bounds[1])
                        if np.min(d) <= field_bounds[0]:
                            end_index = np.argmax(d < field_bounds[0])
                        else:
                            end_index = len(d)
                max_summed_distances = self.firing_fields.max_summed_distances(end_index - start_index)
                self.pass_spreads[field_num][run_num] = \
                    self.firing_fields.spread(~np.isnan(d[start_index:end_index]), max_summed_distances)

                if pass_speeds_from == 'spikes':
                    if self.pass_spike_counts[field_num][run_num] > 1:
                        pass_speed = np.mean(np.array(spike_speeds)[ok_spike_indices])
                        pass_speeds_variations = np.nanstd(np.array(spike_speeds)[ok_spike_indices]) / pass_speed
                    else:
                        pass_speed = np.nan
                        pass_speeds_variations = 0
                elif pass_speeds_from == 'tracking':
                    first_index = np.argmax(d_ok)
                    start_index += first_index
                    end_index += first_index
                    speeds = self.tracking.speed_1D[start_index:end_index] * (1 - 2 * run_type)
                    pass_speed = np.nanmean(speeds)
                    pass_speeds_variations = np.nanstd(speeds) / pass_speed
                else:
                    sys.exit("Parameter 'pass_speed_from' should be 'tracking' or 'spikes'.")
                self.pass_duration[field_num][run_num] = pass_duration
                self.pass_speeds[field_num][run_num] = pass_speed
                self.pass_speeds_variations[field_num][run_num] = pass_speeds_variations

        if only_significant_pp and not os.path.exists(self.pp_ok_fields_path):
            if not os.path.exists(pp_ok_folder):
                os.mkdir(pp_ok_folder)
            print("Calculating fields with significant phase precession...")
            self.significant_precession(num_shuffles=num_shuffles, percentile=percentile)

    def circular_linear_fit(self, positions, phases_in_degrees, place_field_size):
        """Performs a circular linear fit following Schmidt et al., 2009.

        Args:
            positions (np.array): 1D array of positions (cm).
            phases_in_degrees (np.array): 1D array of phases (deg).
            place_field_size (float): Size of the place field (cm).

        Returns:
            (tuple): A tuple containing:
                * slope (float): Fitted slope (deg/cm).
                * intercept (float): Fitted intercept (deg).
        """
        phases = phases_in_degrees/180*np.pi

        def mean_resultant_length(a):
            c = np.sum(np.cos(phases - a[0] * positions))
            s = np.sum(np.sin(phases - a[0] * positions))
            return -math.sqrt((c / positions.size) ** 2 + (s / positions.size) ** 2)

        x0 = np.array([0])
        slope_bounds = [slope_bound*2*np.pi/place_field_size for slope_bound in self.normalized_slope_bounds]
        optimization = optimize.minimize(mean_resultant_length, x0, bounds=[slope_bounds])
        c = np.sum(np.cos(phases - optimization.x * positions))
        s = np.sum(np.sin(phases - optimization.x * positions))
        intercept = np.arctan2(s, c)
        if intercept < np.pi:
            intercept += 2*np.pi
        return (optimization.x/np.pi*180)[0], intercept/np.pi*180

    def circular_orthogonal_fit(self, positions, phases, place_field_size):
        """Calculates a linear fit minimizing the sum of squared errors orthogonal to the fitting line taking into
        account the circular nature of the phase variable. Best fit for line of infinite slope :(

        Args:
            positions (np.array): 1D array of positions (cm).
            phases (np.array): 1D array of phases (deg).
            place_field_size (float): Size of the place field (cm).

        Returns:
            (tuple): A tuple containing:
                * slope (float): Fitted slope (deg/cm).
                * intercept (float): Fitted intercept (deg).
        """
        # normalize positions and phases
        positions /= place_field_size
        phases = phases/360

        # returns sums of errors orthogonal to the fit
        def fit_error(a):
            slope = a[0]
            horizontal_shift = a[1]
            zero_crossings = np.arange(0.5/slope + horizontal_shift, 1 - 1.5/slope, -1/slope)
            normal_vector = np.array((-slope, 1))/math.sqrt(slope**2 + 1)
            orthogonal_errors = np.zeros((zero_crossings.size, positions.size))
            for zero_crossing_num, zero_crossing in enumerate(zero_crossings):
                shifted_positions = positions - zero_crossing
                for point_num, (position, phase) in enumerate(zip(shifted_positions, phases)):
                    orthogonal_errors[zero_crossing_num, point_num] = np.dot(np.array((position, phase)),
                                                                             normal_vector) ** 2
            return np.sum(np.min(orthogonal_errors, axis=0))

        # make a rough search
        num_slopes = 12
        num_shifts = 12
        slopes = np.linspace(self.normalized_slope_bounds[0], self.normalized_slope_bounds[1], num_slopes)
        shifts = np.empty((num_slopes, num_shifts))
        error_sums = np.empty((num_slopes, num_shifts))
        for slope_num, slope in enumerate(slopes):
            shifts[slope_num] = np.linspace(0, -1/slope, num_shifts)
            for shift_num, shift in enumerate(shifts[slope_num]):
                error_sums[slope_num, shift_num] = fit_error([slope, shift])
        best_indices = np.unravel_index(np.argmin(error_sums), error_sums.shape)
        slope = slopes[best_indices[0]]
        horizontal_shift = shifts[best_indices]

        # # optimize around the best parameters, sometimes gives errors
        # x0 = np.array([slope, horizontal_shift])
        # optimization = optimize.minimize(fit_error, x0, bounds=[self.normalized_slope_bounds,
        #                                                         (0, -1/self.normalized_slope_bounds[1])])
        # slope, horizontal_shift = optimization.x

        return slope*360/place_field_size, -slope*(-0.5/slope + horizontal_shift)*360

    @staticmethod
    def lin_model(params, x):
        return params[0] * x + params[1]

    def odr_fit(self, normalized_positions, normalized_phases):
        data = odr.Data(normalized_positions, normalized_phases)
        odr_instance = odr.ODR(data, odr.Model(self.lin_model), beta0=[-1, 1])
        odr_output = odr_instance.run()
        slope = odr_output.beta[0]
        intercept = odr_output.beta[1]
        errors = odr_output.delta ** 2 + odr_output.eps ** 2  # calculate orthogonal errors
        return slope, intercept, errors

    def orthogonal_fit(self, positions, phases, place_field_size, plot_steps=False, fig_name='fitting_steps'):
        """Calculates a linear fit minimizing the sum of squared errors orthogonal to the fitting line. Starting with
        a circular_linear fit, it iteratively attempts to unwrap points far from the fit.

        Args:
            positions (np.array): 1D array of positions (cm).
            phases (np.array): 1D array of phases (deg).
            place_field_size (float): Size of the place field (cm).
            plot_steps (bool): Plot a figure with the intermediate fitting and unwrapping iterations. 
            fig_name (string): Name for the figure. 

        Returns:
            (tuple): A tuple containing:
                * slope (float): Fitted slope (deg/cm).
                * intercept (float): Fitted intercept (deg).
        """
        # normalize positions and phases
        normalized_positions = positions/place_field_size
        normalized_phases = phases / 360
        unwrapped_phases = np.copy(normalized_phases)

        if self.orthogonal_init:
            best_slope, best_intercept, errors = self.odr_fit(normalized_positions, unwrapped_phases)
            best_sum_square_error = np.sum(errors)
        else:
            best_slope = np.nan
            best_intercept = np.nan
            best_sum_square_error = np.inf

        if plot_steps:
            fig, ax = plt.subplots(2, self.num_unwrapping_iterations, sharex='col', sharey='all')
            ax[0, 0].set_ylim([-0.5, 1.5])

        for unwrapping_iteration_num in range(self.num_unwrapping_iterations):
            if self.circular_linear_step and unwrapping_iteration_num == 0:
                # calculate circular linear fit
                slope, intercept = self.circular_linear_fit(positions, phases, place_field_size)
                slope = slope * place_field_size / 360
                intercept /= 360

                # calculate orthogonal errors
                normal_vector = np.array((-slope, 1)) / math.sqrt(slope ** 2 + 1)
                shifted_positions = normalized_positions + intercept / slope
                errors = np.dot(np.vstack((shifted_positions, normalized_phases)).T, normal_vector) ** 2

            else:
                # calculate new orthogonal fit
                slope, intercept, errors = self.odr_fit(normalized_positions, unwrapped_phases)

            sum_square_error = np.sum(errors)

            # update best fit
            if sum_square_error < best_sum_square_error:
                best_sum_square_error = sum_square_error
                best_slope = slope
                best_intercept = intercept

            if plot_steps:
                ax[0, unwrapping_iteration_num].scatter(normalized_positions, unwrapped_phases, marker='.')
                x = np.array((0, max(normalized_positions)))
                ax[0, unwrapping_iteration_num].plot(x, x * slope + intercept, color='C1')
                ax[0, unwrapping_iteration_num].plot(x, x * best_slope + best_intercept, color='C3')
                ax[0, unwrapping_iteration_num].annotate(f"e={np.round(sum_square_error, 2)}", (0.1, 0.1),
                                                         xycoords="axes fraction", fontsize="x-small")

            # move points
            mean_error = np.mean(errors)
            error_threshold = min(mean_error * self.mean_error_threshold_factor, self.min_error_threshold)
            previous_unwrapped_phases = np.copy(unwrapped_phases)
            unwrapped_phases[(errors > error_threshold) & (normalized_positions > self.top_corner[0])
                             & (previous_unwrapped_phases > self.top_corner[1])] -= 1
            unwrapped_phases[(errors > error_threshold) & (normalized_positions < self.bottom_corner[0])
                             & (previous_unwrapped_phases < self.bottom_corner[1])] += 1

            if plot_steps:
                max_error = np.max(errors)
                ax[1, unwrapping_iteration_num].scatter(normalized_positions[errors <= error_threshold],
                                                        previous_unwrapped_phases[errors <= error_threshold],
                                                        c=errors[errors <= error_threshold], marker='.', vmin=0,
                                                        vmax=max_error)
                ax[1, unwrapping_iteration_num].scatter(normalized_positions[errors > error_threshold],
                                                        previous_unwrapped_phases[errors > error_threshold],
                                                        c=errors[errors > error_threshold], marker='X', vmin=0,
                                                        vmax=max_error)
                x = np.array((0, max(normalized_positions)))
                ax[1, unwrapping_iteration_num].plot(x, x * slope + intercept, color='C1')

                self.maybe_save_fig(fig, fig_name, subfolder="/single_runs")

        return float(best_slope * 360 / place_field_size), best_intercept * 360

    def simple_orthogonal_fit(self, positions, phases, place_field_size):
        """Calculates a linear fit minimizing the sum of squared errors orthogonal to the fitting line.
        For points above a certain phase (self.top_corner[1]), it takes the minimum error resulting from either
        (position, phase) or (position, phase - 1). For points below self.bottom_corner[1] it takes the minimum error
        resulting from (position, phase) or (position, phase + 1).

        Args:
            positions (np.array): 1D array of positions (cm).
            phases (np.array): 1D array of phases (deg).
            place_field_size (float): Size of the place field (cm).

        Returns:
            (tuple): A tuple containing:
                * slope (float): Fitted slope (deg/cm).
                * intercept (float): Fitted intercept (deg).
        """
        # normalize positions and phases
        positions /= place_field_size
        phases = phases/360

        # returns sums of errors orthogonal to the fit
        def fit_error(a):
            slope = a[0]
            intercept = a[1]

            shifted_phases = phases - intercept
            normal_vector = np.array((-slope, 1)) / math.sqrt(slope ** 2 + 1)

            orthogonal_errors = np.zeros(positions.size)
            central_indices = (phases >= 0.3) & (phases <= 0.7)
            orthogonal_errors[central_indices] = (np.vstack((positions[central_indices],
                                                            shifted_phases[central_indices])).T @ normal_vector)**2
            high_indices = phases > self.top_corner[1]
            high_points = np.array((np.vstack((positions[high_indices], shifted_phases[high_indices])).T,
                                    np.vstack((positions[high_indices], shifted_phases[high_indices] - 1)).T))
            orthogonal_errors[high_indices] = np.min((high_points @ normal_vector) ** 2, axis=0)

            low_indices = phases < self.bottom_corner[1]
            low_points = np.array((np.vstack((positions[low_indices], shifted_phases[low_indices])).T,
                                   np.vstack((positions[low_indices], shifted_phases[low_indices] + 1)).T))
            orthogonal_errors[low_indices] = np.min((low_points @ normal_vector) ** 2, axis=0)

            return np.sum(orthogonal_errors)

        optimization = optimize.minimize(fit_error, x0=np.array([-1.2, 1.5]), method="Nelder-Mead")
                                         # bounds=[self.normalized_slope_bounds, (-self.normalized_slope_bounds[1],
                                         #                                        1 - self.normalized_slope_bounds[0])])
        slope, intercept = optimization.x
        return slope * 360 / place_field_size, intercept * 360, optimization.fun

    def fit_and_plot(self, positions, phases, run_type, place_field, fit=True, fit_type='orthogonal', min_spikes=2,
                     ax=None, speeds=None, max_speed=None, plot_fitting_steps=False,
                     fitting_steps_fig_name='fitting_steps', normalize_angle_by_size=False, return_sum_error=False):
        """Do a linear fit of phase vs position.

        Args:
            positions (np.array): 1D array of positions (cm).
            phases (np.array): 1D array of phases (deg).
            run_type (int): Forward (0) or backward (1) run.
            place_field (tuple(float)): Spatial extent of the place field (cm).
            fit_type (string): Fit type ("circular_linear", "circular_orthogonal" or "orthogonal").
            min_spikes (int): Minimum number of points to do the fit.
            fit (bool): Whether to perform the fit.
            ax (axes): Axes for plotting.
            speeds (np.array): 1D array of instantaneous spikes for color coding the scatter plot.
            max_speed (float): Maximum speed for setting the scale of the colormap used for coloring spikes. 
            plot_fitting_steps (bool): In the orthogonal fit: plot a figure with the intermediate fitting and 
                unwrapping iterations. 
            fitting_steps_fig_name (string): Name for the figure.
            normalize_angle_by_size (bool): Normalize slope to a field size and phase range of 1 before
                calculating angle.
            return_sum_error (bool): Return sum square error, only possible for simple_orthogonal.
        """
        if run_type == 1:
            positions = place_field[1] - np.array(positions)
        else:
            positions = np.array(positions) - place_field[0]

        place_field_size = place_field[1] - place_field[0]

        if fit and len(positions) > min_spikes:
            if fit_type == "circular_orthogonal":
                slope, intercept = self.circular_orthogonal_fit(np.copy(positions), np.array(phases), place_field_size)
            elif fit_type == "circular_linear":
                slope, intercept = self.circular_linear_fit(positions, phases, place_field_size)
            elif fit_type == "orthogonal":
                slope, intercept = self.orthogonal_fit(np.copy(positions), np.copy(phases), place_field_size,
                                                       plot_steps=plot_fitting_steps, fig_name=fitting_steps_fig_name)
            elif fit_type == "simple_orthogonal":
                slope, intercept, sum_error = \
                    self.simple_orthogonal_fit(np.copy(positions), np.copy(phases), place_field_size)
            else:
                print(fit_type)
                sys.exit("Fit type not recognized")

            slope_for_angle = slope * place_field_size / 360 if normalize_angle_by_size else slope
            angle = np.degrees(np.arctan2(slope_for_angle, 1))

        else:
            slope = np.nan
            intercept = np.nan
            angle = np.nan
            sum_error = np.nan

        # plot
        if ax is not None:
            ax.scatter(positions, phases, c=speeds, vmin=0, vmax=max_speed, s=2, linewidths=0.0)

            if fit_type in ["circular_linear", "circular_orthogonal"]:
                x = np.linspace(0, place_field_size, 100)
                y = x * slope + intercept
                y = y % 360
                jumps = np.diff(y) > 200
                y[:-1][jumps] = np.nan
            else:
                x = np.array((0, place_field_size))
                y = x * slope + intercept

            ax.plot(x, y, color='C1', zorder=5)
            ax.set_xlim([0, place_field_size])
            ax.set_ylim([0, 360])

        if return_sum_error:
            return slope, intercept, angle, sum_error
        else:
            return slope, intercept, angle

    def pool(self, field_num, runs=(), pass_speeds=(), spike_speeds=()):
        """Pool spikes from multiple passes from the field that satisfy certain conditions.

        Args:
            field_num (int): Field number.
            runs (tuple(int)): List of runs to include. Empty for including all.
            pass_speeds (tuple(float)): (min, max) pass speeds to include. Empty tuple includes all.
            spike_speeds (tuple(float)): (min, max) instantaneous spike's speeds to include. Empty tuple includes all.
        """
        positions, phases, speeds = [], [], []

        if not runs:
            runs = range(len(self.positions[field_num]))
        for run_num in runs:
            if (not pass_speeds or
                    (pass_speeds[0] < self.pass_speeds[field_num][run_num] < pass_speeds[1])):
                if not spike_speeds:
                    positions += self.positions[field_num][run_num]
                    phases += self.phases[field_num][run_num]
                    speeds += self.spike_speeds[field_num][run_num]
                else:
                    for spike_num, spike_speed in enumerate(self.spike_speeds[field_num][run_num]):
                        if spike_speeds[0] < spike_speed < spike_speeds[1]:
                            positions += [self.positions[field_num][run_num][spike_num]]
                            phases += [self.phases[field_num][run_num][spike_num]]
                            speeds += [self.spike_speeds[field_num][run_num][spike_num]]

        return np.array(positions), np.array(phases), np.array(speeds)

    def pool_all(self, full_speed_groups, fit_type="orthogonal", min_spikes=10, pool_by_pass_speed=False,
                 spike_speed_threshold=True, min_occupancy=0.4, min_spread=0.4, plot_fits=True, fields_per_plot=6,
                 fig_size=(10, 6), plot_occupancy=False, field_nums=None, constrained_layout=False, within_field=True):
        """For each place field provided, pools spikes from runs based on pass speeds. Calculates fits and plots
        the evolution of the phase precession slope with speed for each cell.

        Args:
            full_speed_groups (list(list(float))): List of (lower, upper) bounds for speed groups. If it includes
                some group including speeds below the minimum pass speed it will be skipped in the analysis and will
                return nans for it.
            fit_type (string): Linear fit type.
            min_spikes (int): Minimum number of points to do the fit.
            pool_by_pass_speed (bool): Pool spikes based on pass speed. If False, pools spikes based on instantaneous
                speeds.
            spike_speed_threshold (bool): When pooling spikes by pass speed, remove spikes below the minimum
                instantaneous speed allowed in the pass speed calculation.
            min_occupancy (float):
            min_spread (float):
            plot_fits (bool): Plot each individual fit.
            fields_per_plot (int): Number of fields per plot.
            fig_size (tuple(float)): Width and height of the figure in inches.
            plot_occupancy (bool):
            field_nums (list(int)): Fields to analyze.
        """
        if pool_by_pass_speed:
            speed_groups = [speed_group for speed_group in full_speed_groups if speed_group[0] >= self.pass_min_speed]
            num_skipped_speed_groups = np.sum([speed_group not in speed_groups for speed_group in full_speed_groups])
        else:
            speed_groups = full_speed_groups
            num_skipped_speed_groups = 0

        max_speed = np.nanmax(self.tracking.speed_1D)
        slopes = []
        angles = []
        mean_errors = []

        occupancy, smooth_occupancy, max_smooth_occupancies = self.firing_fields.occupancy_by_speed(speed_groups)
        fields_spread = self.firing_fields.fields_spread(self.fields['run_types'], self.fields['bound_indices'],
                                                         occupancy, min_occupancy)

        if plot_fits:
            axes = []
            figs = []
            num_fields = len(field_nums) if field_nums is not None else len(self.fields['pair_nums'])
            for plot_num in range(math.ceil(num_fields/fields_per_plot)):
                fig, ax = plt.subplots(len(speed_groups) + 1, fields_per_plot, figsize=fig_size,
                                       constrained_layout=constrained_layout)
                # ax[-2, 0].set_xlabel("Run distance (cm)")
                # ax[-2, 0].set_ylabel("Phase (°)")
                # ax[-1, 0].set_ylabel("Phase\nprecession\nslope (°/cm)")
                # ax[-1, 0].set_xlabel("Running\nspeed (cm/s)")
                axes.append(ax)
                figs.append(fig)

        average_speeds = np.mean(speed_groups, axis=1)
        plotted_field_num = 0
        for field_num, (run_type, field_bound_indices, field_bounds) in \
                enumerate(zip(self.fields['run_types'], self.fields['bound_indices'], self.fields['bounds'])):

            if self.only_significant_pp and not self.fields['pp_ok'][field_num]:
                continue

            if field_nums is not None and field_num not in field_nums:
                continue

            field_slopes = []
            field_angles = []
            field_mean_errors = []

            bound_indices = [field_bound_indices[0], field_bound_indices[1] + 1]

            for speed_group_num, speed_group in enumerate(speed_groups):

                field_smooth_occupancy = smooth_occupancy[run_type, speed_group_num, slice(*bound_indices)]
                if run_type == 1:
                    field_smooth_occupancy = field_smooth_occupancy[::-1]

                occupancy_spread = fields_spread[field_num, speed_group_num]

                if plot_fits:
                    ax = axes[int(plotted_field_num/fields_per_plot)][speed_group_num,
                                                                      plotted_field_num % fields_per_plot]
                    ax.spines['top'].set_visible(False)
                    if speed_group_num == 0:
                        ax.set_title(f"field {field_num}", fontsize='medium')
                    if plotted_field_num % fields_per_plot == 0:
                        ax.annotate(f"{speed_group}\ncm/s", (-1, 0.5), xycoords="axes fraction",
                                    rotation="vertical", va='center', multialignment="center")

                    if speed_group_num < len(speed_groups) - 1:
                        ax.set_xticklabels([])
                    if not plotted_field_num % fields_per_plot == 0:
                        ax.set_yticklabels([])

                    if plot_occupancy:
                        axr = ax.twinx()
                        axr.spines['top'].set_visible(False)
                        axr.tick_params(axis='y', colors='C7')
                        axr.plot(np.arange(len(field_smooth_occupancy))*self.firing_fields.bin_size,
                                 field_smooth_occupancy, color='C7', linewidth=0.6, zorder=1)
                        axr.set_ylim([0, max(max_smooth_occupancies) * 1.1])
                        if not plotted_field_num % fields_per_plot == fields_per_plot - 1:
                            axr.set_yticklabels([])

                        # if (speed_group_num == len(speed_groups) - 1
                        #         and plotted_field_num % fields_per_plot == fields_per_plot - 1):
                            # axr.set_ylabel("Occupancy\n(s)", color='C7')

                        axr.spines['right'].set_edgecolor('C7')
                        # ax.annotate(f"{occupancy_spread:.2f}", (0.75, 0.8), xycoords="axes fraction",
                        #             fontsize="x-small")
                else:
                    ax = None

                if pool_by_pass_speed:
                    pass_speeds = speed_group
                    spike_speed_lower_bound = self.pass_min_speed if spike_speed_threshold else 0
                    spike_speeds = (spike_speed_lower_bound, np.inf)
                else:
                    pass_speeds = ()
                    spike_speeds = speed_group

                positions, phases, speeds = self.pool(field_num, pass_speeds=pass_speeds, spike_speeds=spike_speeds)
                fit = occupancy_spread > min_spread
                slope, _, angle, sum_error = self.fit_and_plot(positions, phases, run_type, field_bounds, fit, fit_type,
                                                               min_spikes, ax, speeds, max_speed, return_sum_error=True)
                field_slopes.append(slope)
                field_angles.append(angle)
                mean_error = sum_error/positions.size if not np.isnan(slope) else np.nan
                field_mean_errors.append(mean_error)

            slopes.append(field_slopes)
            angles.append(field_angles)
            mean_errors.append(field_mean_errors)

            if plot_fits:
                ax = axes[int(plotted_field_num/fields_per_plot)][-1, plotted_field_num % fields_per_plot]
                # ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=1))
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.plot(average_speeds, field_slopes, '.', color='k')
                slopes_ok = ~np.isnan(field_slopes)
                if np.sum(slopes_ok) > 1:
                    fit = linregress(average_speeds[slopes_ok], np.array(field_slopes)[slopes_ok])
                    ax.plot(average_speeds[slopes_ok], average_speeds[slopes_ok] * fit.slope + fit.intercept, 'C7')
                    slopes_range = np.nanmax(field_slopes) - np.nanmin(field_slopes)
                    if slopes_range > 0:
                        ax.set_ylim([np.nanmin(field_slopes) - 0.05*slopes_range,
                                     np.nanmax(field_slopes) + 0.05*slopes_range])

            plotted_field_num += 1

        if plot_fits:
            for fig_num, (fig, ax) in enumerate(zip(figs, axes)):
                if not constrained_layout:
                    fig.tight_layout(h_pad=0.0, w_pad=0.2)
                    fig.subplots_adjust(right=0.8)
                cax = fig.add_axes([0.9, 0.35, 0.012, 0.25])
                color_bar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=max_speed), cmap='viridis'),
                                         cax=cax, aspect=20, shrink=0.3, anchor=(0.0, 0.0))
                color_bar.set_label('Running speed (cm/s)')

                self.maybe_save_fig(fig, f"batch_{fig_num}", subfolder=f"/pooled")

        fig, ax = plt.subplots(2, sharex='col')
        average_speeds = np.mean(speed_groups, axis=1)
        for field_slopes, pair_normalized_angles in zip(slopes, angles):
            ax[0].plot(average_speeds, field_slopes, '-*')
            ax[1].plot(average_speeds, pair_normalized_angles, '-*')
        ax[0].set_ylabel("Phase precession slope (deg/cm)")
        ax[1].set_ylabel("Phase preecssion angle (deg)")
        ax[1].set_xlabel("Running speeds (cm/s)")
        plt.tight_layout()
        self.maybe_save_fig(fig, "slopes_all", subfolder=f"/pooled")

        if num_skipped_speed_groups:
            nans = np.full((len(slopes), num_skipped_speed_groups), np.nan)
            slopes = np.hstack((nans, slopes))
            angles = np.hstack((nans, angles))
            mean_errors = np.hstack((nans, mean_errors))
        else:
            slopes = np.array(slopes)
            angles = np.array(angles)
            mean_errors = np.array(mean_errors)

        self.maybe_pickle_results(slopes, "slopes", subfolder=f"/pooled")
        self.maybe_pickle_results(angles, "angles", subfolder=f"/pooled")
        self.maybe_pickle_results(mean_errors, "mean_errors", subfolder=f"/pooled")

        if within_field:
            self.firing_fields.within_field_increases(speed_groups, self.fields, slopes, "slope_increases",
                                                      subfolder="pooled")

    def single_passes(self, field_num, pass_min_spikes=3, pass_min_duration=0.2, pass_min_spread=0.4,
                      pass_max_variation=0.8, fit_type="orthogonal", plot_fits=False, subplots_per_row=18, fig_width=14,
                      fig_row_height=0.5, fig_extra_height=0.4, plot_slopes=True):
        """Calculate and plot the phase precession slopes for single passes through a place field.
        Plots all the spikes regardless of instantaneous speed, and uses the pass speed calculated from the first and 
        last spike times in the pass through the field. 

        Args:
            field_num (int): Field to analyze.
            pass_min_spikes (int): Minimum number of spikes in the pass.
            pass_min_duration (float): Minimum duration of the pass (s).
            pass_min_spread (float): Minimum measure of the spread of valid tracking positions in the pass.
            pass_max_variation (float): Maximum coefficient of variation for spike's instantaneous speeds.
            fit_type (string): Fit type ("circular_linear", "circular_orthogonal" or "orthogonal")
            plot_fits (bool): Plot individual fits.
            subplots_per_row (int): Number of subplots per row.
            plot_slopes (bool): Plot slopes vs speed.
        """
        pass_speeds = []

        def pass_ok():
            nonlocal field_num, pass_num
            return (self.pass_spike_counts[field_num][pass_num] >= pass_min_spikes
                    and self.pass_duration[field_num][pass_num] >= pass_min_duration
                    and self.pass_spreads[field_num][pass_num] >= pass_min_spread
                    and self.pass_speeds_variations[field_num][pass_num] <= pass_max_variation)

        # find valid runs and sort them by pass speed; find maximum speed.
        ok_pass_indices = []
        for pass_num, pass_speed in enumerate(self.pass_speeds[field_num]):
            if pass_ok():
                ok_pass_indices.append(pass_num)
                pass_speeds.append(pass_speed)

        if len(pass_speeds) == 0:
            return [], [], []

        else:
            ok_pass_indices = np.array(ok_pass_indices)[np.argsort(pass_speeds)]
            pass_speeds = np.sort(pass_speeds)

            slopes = []
            angles = []
            max_speed = np.nanmax(self.tracking.speed_1D)

            # create figure
            if plot_fits:
                num_rows = math.ceil(len(ok_pass_indices)/subplots_per_row)
                fig, ax = plt.subplots(num_rows, subplots_per_row, sharex="col", sharey="row",
                                       figsize=(fig_width, fig_extra_height + num_rows * fig_row_height), squeeze=False)
                # ax[-1, 0].set_xlabel("Run distance (cm)")
                # ax[-1, 0].set_ylabel("Phase (°)")
                color_map = get_cmap('viridis')

            # fit the phase precession slopes and plot clouds and fits
            field_bounds = self.fields['bounds'][field_num]
            for ok_pass_num, (ok_pass_index, pass_speed) in enumerate(zip(ok_pass_indices, pass_speeds)):
                if plot_fits:
                    row_num = int(ok_pass_num / subplots_per_row)
                    axis = ax[row_num, ok_pass_num % subplots_per_row]
                    axis_color = color_map(pass_speed / max_speed)
                    for key in ['top', 'bottom', 'left', 'right']:
                        axis.spines[key].set_color(axis_color)
                    axis.annotate(round(pass_speed, 1), (0.95, 0.75), xycoords="axes fraction", fontsize="small",
                                  horizontalalignment='right')
                    axis.spines['top'].set_visible(False)
                    axis.spines['right'].set_visible(False)
                    # axis.annotate(round(self.pass_speeds_variations[field_num][ok_pass_index], 1), (0.6, 0.6),
                    #               xycoords="axes fraction", fontsize="x-small")
                    # axis.annotate(round(self.pass_spreads[field_num][ok_pass_index], 1), (0.6, 0.4),
                    #               xycoords="axes fraction", fontsize="x-small")
                else:
                    axis = None

                pass_ok_spike_indices = (np.array(self.spike_speeds[field_num][ok_pass_index]) > self.pass_min_speed)
                positions = np.array(self.positions[field_num][ok_pass_index])[pass_ok_spike_indices]
                phases = np.array(self.phases[field_num][ok_pass_index])[pass_ok_spike_indices]
                spike_speeds = np.array(self.spike_speeds[field_num][ok_pass_index])[pass_ok_spike_indices]

                slope, _, normalized_angle = self.fit_and_plot(positions, phases, self.fields['run_types'][field_num],
                                                               field_bounds, fit_type=fit_type, ax=axis,
                                                               speeds=spike_speeds, max_speed=max_speed)
                slopes.append(slope)
                angles.append(normalized_angle)

            if plot_fits:
                # color_bar = fig.colorbar(ScalarMappable(norm=Normalize(vmin=0, vmax=max_speed), cmap='viridis'),
                #                          ax=ax[int(num_rows/2), -1], aspect=20, anchor=(0.0, 0.0))
                # color_bar.set_label("Running speed\n(cm/s)")
                fig.tight_layout(pad=0, h_pad=0.0, w_pad=0.4)
                self.maybe_save_fig(fig, f"field_{field_num}", subfolder=f"/single_runs")

            if plot_slopes:
                fig, ax = plt.subplots(2, sharex='col')
                ax[0].plot(pass_speeds, slopes, '.')
                ax[0].set_ylabel("Phase precession\nslope (°/cm)")
                ax[1].plot(pass_speeds, angles, '.')
                ax[1].set_ylabel("Phase precession\nangle (°)")
                ax[1].set_xlabel("Running speeds (cm/s)")
                fig.align_ylabels()
                plt.tight_layout()
                self.maybe_save_fig(fig, f"slopes_field_{field_num}", subfolder=f"/single_runs")

            return pass_speeds, slopes, angles

    def all_single_passes(self, pass_min_spikes=3, pass_min_duration=0.2, pass_min_spread=0.4, pass_max_variation=0.8,
                          fit_type="orthogonal", plot_fits=False, subplots_per_row=11, plot_slopes=False,
                          percentile=98):
        """For each place field provided, calculates and plots single run phase precession slopes. Displays all slopes
        in one final plot.

        Args:
            pass_min_spikes (int): Minimum number of spikes in the pass.
            pass_min_duration (float): Minimum duration of the pass (s).
            pass_min_spread (float): Minimum measure of the spread of valid tracking positions in the pass.
            pass_max_variation (float): Maximum coefficient of variation for spike's instantaneous speeds.
            fit_type (string): Fit type ("circular_linear", "circular_orthogonal" or "orthogonal")
            plot_fits (bool): Plot individual fits.
            subplots_per_row (int): Number of subplots per row.
            plot_slopes (bool): Plot phase precession slope values.
            percentile (float): Percentile of slopes to show.
        """
        speeds = []
        slopes = []
        angles = []

        for field_num in range(len(self.fields['pair_nums'])):

            if self.only_significant_pp and not self.fields['pp_ok'][field_num]:
                continue

            pair_speeds, pair_slopes, pair_angles = \
                    self.single_passes(field_num, pass_min_spikes, pass_min_duration, pass_min_spread,
                                       pass_max_variation, fit_type, plot_fits=plot_fits,
                                       subplots_per_row=subplots_per_row, plot_slopes=plot_slopes)
            speeds.append(pair_speeds)
            slopes.append(pair_slopes)
            angles.append(pair_angles)

        self.maybe_pickle_results(speeds, "speeds", subfolder=f"/single_runs")
        self.maybe_pickle_results(slopes, "slopes", subfolder=f"/single_runs")
        self.maybe_pickle_results(angles, "angles", subfolder=f"/single_runs")

        if len(np.concatenate(speeds)):
            fig, ax = plt.subplots(2, 1, sharex='col', sharey='row',
                                   # gridspec_kw={'width_ratios': (1, 0.3)},
                                   squeeze=False)
            for pair_speeds, pair_slopes, pair_angles in zip(speeds, slopes, angles):
                ax[0, 0].plot(pair_speeds, pair_slopes, '.')
                ax[1, 0].plot(pair_speeds, pair_angles, '.')
            ax[0, 0].set_ylabel("Phase precession\nslope (deg/cm)")
            ax[1, 0].set_ylabel("Phase precession\nangle (deg)")
            ax[1, 0].set_xlabel("Running speed (cm/s)")
            # ax[0, 1].set_axis_off()
            # ax[1, 1].hist(np.concatenate(angles), orientation='horizontal')
            # ax[1, 1].set_xlabel('Count')
            fig.align_ylabels()
            plt.tight_layout()
            self.maybe_save_fig(fig, f"slopes_all", subfolder=f"/single_runs")

            ax[0, 0].set_ylim([-np.percentile(-np.concatenate(slopes), percentile), 0])
            self.maybe_save_fig(fig, f"slopes_all_zoom", subfolder=f"/single_runs")

    def slopes_vs_stuff(self, fit_type='orthogonal', min_spikes=10, plot_fits=True, subplots_per_row=5, row_height=1.7,
                        extra_height=1, fig_width=10, summary_fig_size=(10, 6)):

        if plot_fits:
            num_rows = math.ceil(len(self.fields['pair_nums']) / subplots_per_row)
            fig, ax = plt.subplots(num_rows, subplots_per_row, sharey="all",
                                   figsize=(fig_width, num_rows*row_height+extra_height), squeeze=False)
            ax[-1, 0].set_ylabel("Phase (deg)")
            ax[-1, 0].set_xlabel("Position (cm)")
        else:
            ax = None

        slopes = []
        angles = []
        spike_mean_speeds = []
        characteristic_speeds = []
        peak_distances_from_start = []
        peak_normalized_pos = []
        peak_distances_to_border = []
        indices = []

        max_speed = np.nanmax(self.tracking.speed_1D)
        for field_num, (run_type, field_bounds) in enumerate(zip(self.fields['run_types'], self.fields['bounds'])):

            if self.only_significant_pp and not self.fields['pp_ok'][field_num]:
                continue

            positions, phases, speeds = self.pool(field_num)
            axis = ax[int(field_num / subplots_per_row), field_num % subplots_per_row]
            slope, _, angle = self.fit_and_plot(positions, phases, run_type, field_bounds, fit_type=fit_type,
                                                min_spikes=min_spikes, ax=axis, speeds=speeds, max_speed=max_speed)

            slopes.append(slope)
            angles.append(angle)
            spike_mean_speeds.append(self.fields['spikes_mean_speeds'][field_num])
            characteristic_speeds.append(self.fields['characteristic_speeds'][field_num])
            peak_distances_from_start.append(self.fields['distances_from_start'][field_num])
            peak_normalized_pos.append(peak_distances_from_start[-1] / self.tracking.d_runs_span)
            peak_distances_to_border.append(self.fields['distances_to_border'][field_num])
            indices.append(self.fields['idx'][field_num])

        if plot_fits:
            # set x axis limits to the maximum limit
            x_lim_max = 0
            for ax_row in ax:
                for axis in ax_row:
                    x_lim = axis.get_xlim()[1]
                    if x_lim > x_lim_max:
                        x_lim_max = x_lim
            for ax_row in ax:
                for axis in ax_row:
                    axis.set_xlim([0, x_lim_max])

            fig.tight_layout()
            self.maybe_save_fig(fig, "fits", subfolder=f"/all_spikes")

        self.maybe_pickle_results([slopes], "slopes", subfolder=f"/all_spikes")
        self.maybe_pickle_results([angles], "angles", subfolder=f"/all_spikes")
        self.maybe_pickle_results([spike_mean_speeds], "spike_mean_speeds", subfolder=f"/all_spikes")
        self.maybe_pickle_results([characteristic_speeds], "characteristic_speeds", subfolder=f"/all_spikes")
        self.maybe_pickle_results([peak_distances_from_start], "peak_distances_from_start", subfolder=f"/all_spikes")
        self.maybe_pickle_results([peak_normalized_pos], "peak_normalized_pos", subfolder=f"/all_spikes")
        self.maybe_pickle_results([peak_distances_to_border], "peak_distances_to_border", subfolder=f"/all_spikes")
        self.maybe_pickle_results([indices], "indices", subfolder=f"/all_spikes")

        fig, ax = plt.subplots(2, 4, figsize=summary_fig_size)
        ax[0, 0].scatter(peak_distances_from_start, slopes)
        ax[0, 0].set_ylabel("Phase precession\nslope (deg/cm)")
        ax[0, 1].scatter(peak_distances_to_border, slopes)
        ax[0, 2].scatter(spike_mean_speeds, slopes)
        ax[0, 3].scatter(characteristic_speeds, slopes)
        ax[1, 0].scatter(peak_distances_from_start, angles)
        ax[1, 0].set_ylabel("Phase precession\nangle (deg)")
        ax[1, 0].set_xlabel("Distance from\nthe start of the run (cm)")
        ax[1, 1].scatter(peak_distances_to_border, angles)
        ax[1, 1].set_xlabel("Distance to\nthe nearest border (cm)")
        ax[1, 2].scatter(spike_mean_speeds, angles)
        ax[1, 2].set_xlabel("Mean speed\nfor field's spikes (cm/s)")
        ax[1, 3].scatter(characteristic_speeds, angles)
        ax[1, 3].set_xlabel("Characteristic speed\nthrough the field (cm/s)")
        self.maybe_save_fig(fig, "slopes_vs_stuff", subfolder=f"/all_spikes")

    @staticmethod
    def max_summed_distances(pattern_size):
        sum_distance = 0
        for i in range(pattern_size):
            for j in range(i + 1, pattern_size):
                sum_distance += abs(i - j)
        return sum_distance

    def find_theta_0(self, min_shift, max_shift, num_shifts, min_spikes=10):
        shifts = np.linspace(min_shift, max_shift, num_shifts)
        all_mean_errors = []
        for shift in shifts:
            mean_errors = []
            for field_num, (run_type, field_bounds) in enumerate(zip(self.fields['run_types'], self.fields['bounds'])):
                # calculate mean error for the orthogonal fit
                place_field_size = field_bounds[1] - field_bounds[0]

                positions = []
                phases = []
                for run_num in range(len(self.positions[field_num])):
                    positions += self.positions[field_num][run_num]
                    phases += self.phases[field_num][run_num]

                if len(positions) > min_spikes:
                    normalized_positions = np.array(positions) / place_field_size
                    normalized_phases = ((np.array(phases) + shift) % 360) / 360

                    slope, intercept, errors = self.odr_fit(normalized_positions, normalized_phases)
                    mean_errors.append(np.mean(errors))
            all_mean_errors.append(np.mean(mean_errors))

        fig, ax = plt.subplots()
        ax.plot(shifts, all_mean_errors, '.-')
        ax.set_ylabel("Mean error")
        ax.set_xlabel("Phase shift (deg)")
        self.maybe_save_fig(fig, "phase_shifts")

        best_shift = shifts[np.argmin(all_mean_errors)]

        with open(f"sessions/{self.super_group_name}.json", 'r') as f:
            session_dict = json.load(f)
        session_dict['phase_shift'] = best_shift
        with open(f"sessions/{self.super_group_name}.json", 'w') as f:
            json.dump(session_dict, f, indent=2)

        print(f"Best phase shift: {best_shift}")

    def significant_precession(self, num_shuffles=100, percentile=5, plot=True, fig_size=(5.5, 3)):
        self.fields['pp_ok'] = []
        for field_num, (run_type, field_bounds) in enumerate(zip(self.fields['run_types'], self.fields['bounds'])):
            field_positions, field_phases, _ = self.pool(field_num, spike_speeds=(2, np.inf))
            if run_type == 1:
                field_positions = field_bounds[1] - field_positions
            else:
                field_positions = field_positions - field_bounds[0]
            field_size = field_bounds[1] - field_bounds[0]

            data_slope, data_intercept, data_sum_error = self.simple_orthogonal_fit(field_positions.copy(),
                                                                                    field_phases, field_size)
            data_mean_error = data_sum_error / len(field_positions)

            shuffled_sum_errors = []
            for shuffle_num in range(num_shuffles):
                permuted_phases = field_phases[np.random.permutation(len(field_phases))].copy()
                slope, intercept, sum_error = self.simple_orthogonal_fit(field_positions.copy(), permuted_phases,
                                                                         field_size)
                shuffled_sum_errors.append(sum_error)

            shuffled_mean_errors = np.array(shuffled_sum_errors) / len(field_positions)
            pp_ok = data_mean_error < np.percentile(shuffled_mean_errors, percentile)
            self.fields['pp_ok'].append(pp_ok)

            if plot:
                fig, ax = plt.subplots(1, 2, figsize=fig_size)
                ax[0].plot(field_positions, field_phases, '.')
                x = np.array((0, field_size))
                ax[0].plot(x, x*data_slope + data_intercept)
                ax[0].set_ylim((0, 360))
                ax[0].set_ylabel("Phase (deg)")
                ax[0].set_xlabel("Position (cm)")

                ax[1].hist(shuffled_mean_errors)
                ax[1].axvline(data_mean_error, color='red')
                ax[1].set_ylabel("Count")
                ax[1].set_xlabel("Mean orthogonal error")

                plt.tight_layout()
                self.maybe_save_fig(fig, f"{'s' if pp_ok else 'ns'}_field_{field_num}", subfolder="pp_ok")

        with open(self.pp_ok_fields_path, 'wb') as fields_file:
            pickle.dump(self.fields, fields_file)

    def curvature_by_acceleration(self, num_means=3, positions_percentile=90, plot=False, plots_per_row=10,
                                  row_height=1.5, width=12):
        speeds = []
        accelerations = []
        c_speed_slopes = []
        rel_c_speed_changes = []
        positions = []
        phases = []
        circular_means = []
        mean_ranges = []
        curvatures = []

        for field_num, (run_type, field_bound_indices, field_bounds) in \
                enumerate(zip(self.fields['run_types'], self.fields['bound_indices'], self.fields['bounds'])):

            if self.only_significant_pp and not self.fields['pp_ok'][field_num]:
                continue

            lower_bound, upper_bound = field_bound_indices
            field_speeds = self.tracking.characteristic_speeds[run_type][lower_bound:upper_bound + 1]
            if run_type == 1:
                field_speeds = field_speeds[::-1]
            speeds.append(field_speeds)

            accelerations.append(np.mean(self.tracking.mean_acceleration[run_type][lower_bound:upper_bound + 1]))

            not_nan = ~np.isnan(field_speeds)
            x = (np.arange(len(field_speeds))*self.firing_fields.bin_size)[not_nan]
            slope = linregress(x, field_speeds[not_nan])[0]
            c_speed_slopes.append(slope)

            rel_c_speed_changes.append(slope * len(field_speeds) * self.firing_fields.bin_size
                                       / np.nanmean(field_speeds))

            field_positions, field_phases, _ = self.pool(field_num, spike_speeds=(2, np.inf))

            if run_type == 1:
                field_positions = field_bounds[1] - field_positions
            else:
                field_positions = field_positions - field_bounds[0]
            field_size = field_bounds[1] - field_bounds[0]
            field_positions /= field_size
            positions.append(field_positions)
            phases.append(field_phases)

            field_circular_means = []
            field_mean_ranges = []

            start_dense = np.percentile(field_positions, 50 - positions_percentile / 2)
            end_dense = np.percentile(field_positions, 50 + positions_percentile / 2)
            dense_span = end_dense - start_dense

            for mean_num in range(num_means):
                start = start_dense + mean_num / num_means * dense_span
                end = start_dense + (mean_num + 1) / num_means * dense_span
                bound_phases = field_phases[(start < field_positions) & (field_positions < end)] / 180 * np.pi
                field_circular_mean = np.arctan2(np.sum(np.sin(bound_phases)), np.sum(np.cos(bound_phases)))
                field_circular_means.append(field_circular_mean / np.pi * 180 % 360)
                field_mean_ranges.append((start, end))
            circular_means.append(field_circular_means)
            mean_ranges.append(field_mean_ranges)

            diff = np.diff(field_circular_means)
            if (diff > 0).any():
                curvatures.append(np.nan)
            else:
                curvatures.append(np.mean(np.diff(diff)))

        self.maybe_pickle_results(speeds, "speeds", subfolder=f"/acceleration")
        self.maybe_pickle_results([accelerations], "accelerations", subfolder=f"/acceleration")
        self.maybe_pickle_results([c_speed_slopes], "c_speed_slopes", subfolder=f"/acceleration")
        self.maybe_pickle_results([rel_c_speed_changes], "rel_c_speed_changes", subfolder=f"/acceleration")
        self.maybe_pickle_results(positions, "positions", subfolder=f"/acceleration")
        self.maybe_pickle_results(phases, "phases", subfolder=f"/acceleration")
        self.maybe_pickle_results(circular_means, "circular_means", subfolder=f"/acceleration")
        self.maybe_pickle_results(mean_ranges, "mean_ranges", subfolder=f"/acceleration")
        self.maybe_pickle_results([curvatures], "curvatures", subfolder=f"/acceleration")

        if plot:
            rows = int(np.ceil(len(accelerations)/plots_per_row))
            fig, ax = plt.subplots(rows, plots_per_row, sharey='all', sharex='all', squeeze=False,
                                   figsize=(width, rows*row_height))
            sorted_indices = np.argsort(accelerations)
            for field_num, sorted_index in enumerate(sorted_indices):
                axis = ax[int(field_num/plots_per_row), field_num % plots_per_row]
                for mean_range in mean_ranges[sorted_index]:
                    axis.axvline(mean_range[0], color='C7')
                    axis.axvline(mean_range[1], color='C7')
                axis.plot(positions[sorted_index], phases[sorted_index], '.', alpha=0.1)
                axis.set_title(f"{accelerations[sorted_index]:.2f} "r"cm/s$^2$", fontsize='small')
                x = [(mean_range[0] + mean_range[1])/2 for mean_range in mean_ranges[sorted_index]]
                axis.plot(x, circular_means[sorted_index], '*-', color='k')

                axis.annotate(f"{curvatures[sorted_index]:.2f}", xy=(0.05, 0.1), xycoords='axes fraction',
                              fontsize='x-small')
            fig.tight_layout()
            self.maybe_save_fig(fig, "cloud_fits", subfolder="acceleration")

            fig, ax = plt.subplots(2, sharey='row', figsize=(3.5, 6))
            ax[0].plot(accelerations, curvatures, '.')
            ax[0].axhline(0, color='C7', linestyle='dashed')
            ax[0].axvline(0, color='C7', linestyle='dashed')

            ax[0].set_ylabel("Curvature")
            ax[0].set_xlabel(r"Acceleration cm/s$^2$")
            ax[1].plot(c_speed_slopes, curvatures, '.')
            ax[1].axhline(0, color='C7', linestyle='dashed')
            ax[1].axvline(0, color='C7', linestyle='dashed')
            ax[1].set_ylabel("Curvature")
            ax[1].set_xlabel("Characteristic speed slope")
            fig.tight_layout()

            self.maybe_save_fig(fig, "curvatures", subfolder="acceleration")

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):

        return cls(super_group_name, group_name, child_name, kwargs['LFP'], kwargs['FiringFields'],
                   parameters_dict['only_significant_pp'], parameters_dict['num_shuffles'],
                   parameters_dict['percentile'],
                   parameters_dict['normalized_slope_bounds'], parameters_dict['pass_min_speed'],
                   parameters_dict['pass_speeds_from'], parameters_dict['orthogonal_fit_params'],
                   save_figures=save_figures, figure_format=figure_format, figures_path=figures_path,
                   pickle_results=pickle_results, pickles_path=pickles_path)

