import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from data_analysis.lfp import LFP
from data_analysis.tracking import Tracking
from data_analysis.spikes_basics import SpikesBase, load_spikes


class ExpSpikes(SpikesBase):
    """Class for loading and holding spiking data from the datasets published by Mizuseki et al., 2014.
    The folder structure and file names of the dataset must be left unaltered for this class to work properly.

    Args:
        data_path (string): Path to the data.
        dataset (string): Name of the dataset, e.g., hc-3.
        session_set (string): Name of the set of sessions that get lumped together, e.g., ec013.28.
        session (string): Name of the session, e.g., ec013.412.
        region (string): Brain region in which the electrodes were placed (as per the metadata table).
        cell_type (string): Cell type (as by the metadata table).
        diba_clusters (bool): Use Kamran Diba's clustering.
    """

    dependencies = (LFP,)

    def __init__(self, super_group_name, group_name, child_name, data_path, dataset, session_set, session,
                 discarded_intervals=(), region='CA1', cell_type='p', diba_clusters=False):

        super().__init__(super_group_name, group_name, child_name)
        self.spikes = load_spikes(data_path, dataset, session_set, session, discarded_intervals, region,
                                  cell_type, diba_clusters)

    def discard_no_theta(self, lfp):
        """Discard spikes that occur during periods without significant theta oscillations.

        Args:
            lfp (LFP): LFP instance.
        """
        for pair_num, electrode_cluster_pair in enumerate(self.spikes.electrode_cluster_pairs):
            electrode_index = self.spikes.electrodes.index(electrode_cluster_pair[0])
            clean_spikes = []
            for spike_time in self.spikes.spike_times[pair_num]:
                if lfp.at_time(spike_time, electrode_index, return_significance=True)[0]:
                    clean_spikes.append(spike_time)
            self.spikes.spike_times[pair_num] = np.array(clean_spikes)

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):
        spikes = cls(super_group_name, group_name, child_name, kwargs['data_path'],
                     parameters_dict['dataset'], parameters_dict['session_set'], parameters_dict['session'],
                     parameters_dict['discarded_intervals'], parameters_dict['region'], parameters_dict['cell_type'])
        spikes.discard_no_theta(kwargs['LFP'])
        return spikes


class ModelSpikes(SpikesBase):

    dependencies = (LFP, Tracking)

    def __init__(self, super_group_name, group_name, child_name, lfp, tracking, num_cells, ds, dt,
                 phase_range, phase_current, firing_rate_0, firing_rage_slope, theta_modulation, field_centers=None,
                 save_figures=False, figure_format="png", figures_path="figures"):

        super().__init__(super_group_name, group_name, child_name, save_figures, figure_format, figures_path)

        self.lfp = lfp
        self.tracking = tracking
        self.num_cells = num_cells
        self.spikes.spike_times = [[] for _ in range(num_cells)]
        self.spikes.electrodes = [0]
        self.spikes.electrode_cluster_pairs = [[0, field_num] for field_num in range(num_cells)]

        self.ds = ds
        self.track_length = np.nanmax(tracking.d)
        self.num_spatial_bins = int(round(self.track_length/self.ds) + 1)
        if field_centers is None:
            self.field_centers = np.linspace(self.tracking.d_runs_offset,
                                             self.tracking.d_runs_offset + self.tracking.d_runs_span, num_cells)
        elif type(field_centers) == list and len(field_centers) == num_cells:
            self.field_centers = field_centers
        else:
            sys.exit("field_centers must be None or match the number of cells!")

        self.fields = np.zeros((2, self.num_spatial_bins, num_cells))
        self.field_sigmas = None
        self.field_activations = None

        self.dt = dt
        self.phase_range = phase_range
        self.phase_range_extent = self.phase_range[1] - self.phase_range[0]
        self.phase_current = phase_current
        self.firing_rate_0 = firing_rate_0
        self.firing_rate_slope = firing_rage_slope
        self.theta_modulation = theta_modulation

        self.times = None
        self.positions = None
        self.theta_phases = None

    def define_fields(self):
        x = np.arange(0, self.num_spatial_bins)
        for run_type in range(2):
            for field_num, (field_center, field_sigma) in enumerate(zip(self.field_centers, self.field_sigmas[run_type])):
                center = field_center/self.ds
                field_sigma_squared = (field_sigma / self.ds) ** 2
                self.fields[run_type, :, field_num] = np.exp(-np.square(x - center) / 2 / field_sigma_squared)

    def before_loop(self):
        pass

    def inside_loop(self, time_step, time, run_type, speed):
        """Must compute self.field_activations.
        """
        pass

    def generate_spikes(self, time_interval=(), lfp_channel_index=0):
        if time_interval:
            self.times = np.arange(time_interval[0], time_interval[1] + self.dt, self.dt)
        else:
            self.times = np.arange(self.tracking.times[0], min(self.tracking.times[-1], self.lfp.times[-1]), self.dt)
        self.positions = np.empty(self.times.size)
        self.theta_phases = np.empty(self.times.size)

        self.before_loop()

        for time_step, time in enumerate(self.times):
            if time % 100 == 0:
                print(f"generating spikes at time {int(time)}s (/{self.times[-1]}s)")

            self.positions[time_step], run_type, speed, significant_theta = \
                self.tracking.at_time(time, return_speed=True, return_significant_theta=True)

            self.theta_phases[time_step], = self.lfp.at_time(time, lfp_channel_index, return_phase=True)

            if run_type != -1 and significant_theta and not np.isnan(speed):
                if self.phase_range[0] <= self.theta_phases[time_step] <= self.phase_range[1]:
                    self.inside_loop(time_step, time, run_type, speed)
                    if self.field_activations is not None:
                        firing_rate = self.firing_rate_0 + self.firing_rate_slope * speed
                        for field_index in np.nonzero(np.random.random(self.num_cells) <
                                                      firing_rate * self.dt * self.field_activations)[0]:
                            self.spikes.spike_times[field_index].append(time)

    def plot(self, time_interval):
        first_index = int((time_interval[0] - self.times[0]) / self.dt)
        last_index = int((time_interval[1] - self.times[0]) / self.dt)
        times = self.times[first_index:last_index]

        fig, ax = plt.subplots(2, 3, figsize=(9, 6), sharex='col', sharey='row',
                               gridspec_kw={'height_ratios': [4, 1], 'width_ratios': [6, 1, 1]}, num='trajectories')
        ax[1, 1].axis('off')
        ax[1, 2].axis('off')

        # plot trajectory
        ax[0, 0].set_ylabel('Position (m)')
        ax[0, 0].plot(times, self.positions[first_index:last_index], 'k', label='actual path')

        self.plot_extra(ax, first_index, last_index, times)

        # plot spikes
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for field_num, field_spike_times in enumerate(self.spikes.spike_times):
            field_spike_times = np.array(field_spike_times)
            valid_spike_times = (field_spike_times[(field_spike_times > time_interval[0])
                                                   & (field_spike_times < time_interval[1])])
            ax[0, 0].plot(valid_spike_times, np.ones(len(valid_spike_times))*self.field_centers[field_num], '|',
                          markeredgewidth=1.5, color=colors[field_num % len(colors)])
        ax[0, 0].legend(loc='lower right')

        # plot place fields
        for run_type in range(2):
            for field_num in range(self.fields.shape[-1]):
                ax[0, run_type + 1].plot(self.fields[run_type, :, field_num],
                                         np.arange(0, self.num_spatial_bins) * self.ds)
                ax[0, run_type + 1].set_xlabel(f"'True'\nplace fields\n{'^' if run_type == 0 else 'v'}")

        # plot theta oscillation
        ax[1, 0].set_xlabel('Time (s)')
        ax[1, 0].set_ylabel('Theta\nphase\n(deg)')
        ax[1, 0].plot(times, self.theta_phases[first_index:last_index], 'k')

        fig.align_ylabels()
        plt.tight_layout()

        self.maybe_save_fig(fig, "spike_generation")

    def plot_extra(self, ax, first_index, last_index, times):
        pass


class UniformSpikes(ModelSpikes):
    def __init__(self, super_group_name, group_name, child_name, lfp, tracking, num_cells, field_sigma, ds, dt,
                 phase_range, phase_current, theta_time, theta_distance, firing_rate_0, firing_rage_slope,
                 theta_modulation, save_figures=False, figure_format="png", figures_path="figures"):

        super().__init__(super_group_name, group_name, child_name, lfp, tracking, num_cells, ds, dt, phase_range,
                         phase_current, firing_rate_0, firing_rage_slope, theta_modulation, save_figures=save_figures,
                         figure_format=figure_format, figures_path=figures_path)

        self.theta_paths = None
        self.theta_time = theta_time
        self.theta_distance = theta_distance
        self.field_sigmas = np.ones((2, self.num_cells)) * field_sigma
        self.define_fields()

    def before_loop(self):
        self.theta_paths = np.full(self.times.size, np.nan)

    def inside_loop(self, time_step, time, run_type, speed):
        relative_phase = (self.theta_phases[time_step] - self.phase_current) / self.phase_range_extent
        travelling_time = time + relative_phase * self.theta_time
        travelling_time = max(min(travelling_time, self.tracking.times[-1]), 0)
        self.theta_paths[time_step] = self.tracking.at_time(travelling_time)[0]
        if self.theta_distance != 0 and run_type == 0:
            self.theta_paths[time_step] += relative_phase * self.theta_distance
        elif self.theta_distance != 0:  # (and run_type == 1)
            self.theta_paths[time_step] -= relative_phase * self.theta_distance

        # calculate field activations
        self.field_activations = None
        if not np.isnan(self.theta_paths[time_step]):
            spatial_bin = int(round(self.theta_paths[time_step] / self.ds))
            if 0 <= spatial_bin < self.num_spatial_bins:
                theta = np.cos(self.theta_phases[time_step] / 180 * np.pi)
                self.field_activations = self.fields[run_type, spatial_bin] * (
                        (-theta + 1) / 2 * self.theta_modulation + (1 - self.theta_modulation))

    def plot_extra(self, ax, first_index, last_index, times):
        # plot theta paths
        nan_threshold = 180
        nan_indices = np.nonzero(np.abs(np.diff(self.theta_phases[first_index:last_index])) > nan_threshold)[0] + 1
        time_points = np.insert(times, nan_indices, np.nan)
        theta_paths = np.insert(self.theta_paths[first_index:last_index], nan_indices, np.nan)
        ax[0, 0].plot(time_points, theta_paths, 'C7', label='theta paths')
        ax[0, 0].legend(loc="lower right")

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):
        model_spikes = cls(super_group_name, group_name, child_name, kwargs['LFP'], kwargs['Tracking'],
                           parameters_dict["num_cells"], parameters_dict["field_sigma"], parameters_dict["ds"],
                           parameters_dict["dt"], parameters_dict["phase_range"], parameters_dict["phase_current"],
                           parameters_dict["theta_time"], parameters_dict["theta_distance"],
                           parameters_dict["firing_rate_0"], parameters_dict["firing_rate_slope"],
                           parameters_dict["theta_modulation"], save_figures, figure_format, figures_path)

        model_spikes.generate_spikes()
        return model_spikes


class VariableSpikes(ModelSpikes):
    """Each field has its own fixed theta distance based on the characteristic speed through the field.
    Also OK for variable noise model.
    """
    def __init__(self, super_group_name, group_name, child_name, lfp, tracking, num_cells, ds, dt,
                 phase_range, phase_current, firing_rate_0, firing_rage_slope, theta_modulation, phase_sigma_0,
                 exponential_factor, save_figures=False, figure_format="png", figures_path="figures"):

        super().__init__(super_group_name, group_name, child_name, lfp, tracking, num_cells, ds, dt, phase_range,
                         phase_current, firing_rate_0, firing_rage_slope, theta_modulation, save_figures=save_figures,
                         figure_format=figure_format, figures_path=figures_path)

        self.theta_distances = []
        self.field_sigmas = []
        self.phase_sigma_0 = phase_sigma_0
        self.exponential_factor = exponential_factor

    def field_sigmas_from_theta_d(self, size_to_theta_d, size_min, size_sigma):
        return np.maximum(size_min, self.theta_distances * size_to_theta_d
                          + np.random.normal(0, size_sigma, size=(2, self.num_cells)))

    def plot_field_params(self, x_values, x_label, name):
        fig, ax = plt.subplots(2)
        for run_type, run_type_name in enumerate(self.tracking.run_type_names):
            ax[0].plot(x_values[run_type], self.theta_distances[run_type], 'o', label=run_type_name.lower())
            ax[1].plot(x_values[run_type], self.field_sigmas[run_type], 'o', label=run_type_name.lower())
        ax[0].set_ylabel("Theta distance (cm)")
        ax[1].set_ylabel("Field sigma (cm)")
        ax[1].set_xlabel(x_label)
        ax[0].legend()
        ax[1].legend()
        self.maybe_save_fig(fig, name)

    def fields_from_characteristic_speed(self, theta_d_slope, theta_d_offset, theta_d_min, theta_d_add_sigma,
                                         theta_d_mul_sigma, size_to_theta_d, size_min, size_add_sigma, plot=True):

        selected_mean_speeds = \
            self.tracking.characteristic_speeds[:, np.round((self.field_centers - self.tracking.d_runs_offset)
                                                            / self.tracking.spatial_bin_size).astype(int)]
        self.theta_distances = np.maximum((theta_d_offset + theta_d_slope * selected_mean_speeds)
                                          * np.random.normal(1, theta_d_mul_sigma, size=(2, self.num_cells))
                                          + np.random.normal(0, theta_d_add_sigma, size=(2, self.num_cells)),
                                          theta_d_min)

        self.field_sigmas = self.field_sigmas_from_theta_d(size_to_theta_d, size_min, size_add_sigma)
        self.define_fields()

        if plot:
            self.plot_field_params(selected_mean_speeds, "Characteristic speed (cm/s)", "speed_field_params")

    def fields_from_arc(self, theta_d_offset, theta_d_peak, theta_d_min, theta_d_sigma, size_to_theta_d, size_min,
                        size_sigma, plot=True):
        r = self.tracking.d_runs_span/2
        y = np.sqrt(r ** 2 - (self.field_centers - self.tracking.d_runs_offset - r) ** 2)
        self.theta_distances = np.maximum(theta_d_min, y/r*(theta_d_peak - theta_d_offset) + theta_d_offset
                                          + np.random.normal(0, theta_d_sigma, size=(2, self.num_cells)))

        self.field_sigmas = self.field_sigmas_from_theta_d(size_to_theta_d, size_min, size_sigma)
        self.define_fields()

        if plot:
            self.plot_field_params(np.stack((self.field_centers, self.field_centers)), "Field peak position (cm)",
                                   "arc_field_params")

    def fields_from_sigmoid(self, steepness, inflection_point, theta_d_offset, theta_d_peak, theta_d_min, theta_d_sigma,
                            size_to_theta_d, size_min, size_sigma, plot=True):

        relative_field_centers = self.field_centers - self.tracking.d_runs_offset
        x = np.where(relative_field_centers > self.tracking.d_runs_span/2,
                     self.tracking.d_runs_span - relative_field_centers, relative_field_centers)
        y = 1/(1 + np.exp(-steepness*(x - self.tracking.d_runs_span * inflection_point)))
        self.theta_distances = np.maximum(theta_d_min, (y*(theta_d_peak - theta_d_offset) + theta_d_offset)
                                          + np.random.normal(0, theta_d_sigma, size=(2, self.num_cells)))

        self.field_sigmas = self.field_sigmas_from_theta_d(size_to_theta_d, size_min, size_sigma)
        self.define_fields()

        if plot:
            self.plot_field_params(np.stack((self.field_centers, self.field_centers)), "Field peak position (cm)",
                                   "sigmoid_field_params")

    def fields_from_parabola(self, theta_d_offset, theta_d_peak, theta_d_min, theta_d_sigma, size_to_theta_d, size_min,
                             size_sigma, plot=True):

        relative_field_centers = self.field_centers - self.tracking.d_runs_offset
        a = (theta_d_peak - theta_d_offset) * 4 / (self.tracking.d_runs_span**2)
        self.theta_distances = np.maximum(theta_d_min, -a*(relative_field_centers - self.tracking.d_runs_span/2)**2
                                          + theta_d_peak + np.random.normal(0, theta_d_sigma, size=(2, self.num_cells)))
        self.field_sigmas = self.field_sigmas_from_theta_d(size_to_theta_d, size_min, size_sigma)
        self.define_fields()

        if plot:
            self.plot_field_params(np.stack((self.field_centers, self.field_centers)), "Field peak position (cm)",
                                   "parabola_field_params")

    def inside_loop(self, time_step, time, run_type, speed):
        if self.phase_sigma_0:
            theta_phase = np.random.normal(self.theta_phases[time_step],
                                           self.phase_sigma_0 * math.exp(self.exponential_factor * speed),
                                           self.num_cells) % 360
        else:
            theta_phase = self.theta_phases[time_step]

        relative_phase = (theta_phase - self.phase_current) / self.phase_range_extent

        represented_positions = (self.positions[time_step]
                                 + (1 - 2 * run_type) * relative_phase * self.theta_distances[run_type])
        valid_positions = (represented_positions >= 0) & (represented_positions < self.track_length)
        represented_spatial_bins = np.round(np.ma.masked_array(represented_positions, mask=~valid_positions)
                                            / self.ds).astype(int)

        self.field_activations = np.zeros(self.num_cells)
        theta = np.cos(self.theta_phases[time_step] / 180 * np.pi)
        self.field_activations[valid_positions] = (self.fields[run_type,
                                                               represented_spatial_bins[valid_positions],
                                                               valid_positions]
                                                   * ((-theta + 1) / 2 * self.theta_modulation +
                                                      (1 - self.theta_modulation)))

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, p, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):

        if "phase_sigma_0" in p:
            phase_sigma_0 = p['phase_sigma_0']
            exponential_factor = p['exponential_factor']
        else:
            phase_sigma_0 = 0
            exponential_factor = None

        model_spikes = cls(super_group_name, group_name, child_name, kwargs['LFP'], kwargs['Tracking'], p["num_cells"],
                           p["ds"], p["dt"], p["phase_range"], p["phase_current"], p["firing_rate_0"],
                           p["firing_rate_slope"], p["theta_modulation"], phase_sigma_0, exponential_factor,
                           save_figures, figure_format, figures_path)

        if p["fields_from"] == "characteristic_speed":
            model_spikes.fields_from_characteristic_speed(p["theta_d_slope"], p["theta_d_offset"], p["theta_d_min"],
                                                          p["theta_d_add_sigma"], p["theta_d_mul_sigma"],
                                                          p["size_to_theta_d"], p["size_min"], p["size_add_sigma"])
        elif p["fields_from"] == "arc":
            model_spikes.fields_from_arc(p['theta_d_offset'], p['theta_d_peak'], p['theta_d_min'], p['theta_d_sigma'],
                                         p["size_to_theta_d"], p["size_min"], p["size_sigma"])
        elif p["fields_from"] == "sigmoid":
            model_spikes.fields_from_sigmoid(p['steepness'], p['inflection_point'], p['theta_d_offset'],
                                             p['theta_d_peak'], p['theta_d_min'], p['theta_d_sigma'],
                                             p["size_to_theta_d"], p["size_min"], p['size_sigma'])
        elif p["fields_from"] == "parabola":
            model_spikes.fields_from_parabola(p['theta_d_offset'], p['theta_d_peak'], p['theta_d_min'],
                                              p['theta_d_sigma'], p["size_to_theta_d"], p["size_min"], p["size_sigma"])
        else:
            sys.exit("Parameter 'fields_from' lacks a valid assignment.")

        model_spikes.generate_spikes()

        return model_spikes


class SpeedSpikes(ModelSpikes):
    """Represented position depends on characteristic speed at current position (behavior-dependent sweeps).
    """
    def __init__(self, super_group_name, group_name, child_name, lfp, tracking, num_cells, ds, dt,
                 phase_range, phase_current, firing_rate_0, firing_rage_slope, theta_modulation, theta_time,
                 multiplicative_sigma, additive_sigma, size_to_theta_d, size_min, shift_sigma, field_centers=None,
                 save_figures=False, figure_format="png", figures_path="figures"):

        super().__init__(super_group_name, group_name, child_name, lfp, tracking, num_cells, ds, dt, phase_range,
                         phase_current, firing_rate_0, firing_rage_slope, theta_modulation, field_centers, save_figures,
                         figure_format, figures_path)

        self.theta_time = theta_time
        self.multipliers = np.random.normal(1, multiplicative_sigma, size=(2, self.num_cells))
        self.adders = np.random.normal(0, additive_sigma, size=(2, self.num_cells))

        # define fields
        selected_mean_speeds = \
            self.tracking.characteristic_speeds[:, np.round((self.field_centers - self.tracking.d_runs_offset)
                                                            / self.tracking.spatial_bin_size).astype(int)]
        approx_theta_distances = np.maximum((theta_time * selected_mean_speeds) * self.multipliers + self.adders, 0)
        self.field_sigmas = np.maximum(size_min, approx_theta_distances * size_to_theta_d)
        self.define_fields()

        self.phase_shifts = np.random.normal(0, shift_sigma, size=num_cells)

    def inside_loop(self, time_step, time, run_type, speed):
        d_run = self.positions[time_step] - self.tracking.d_runs_offset
        characteristic_speed = self.tracking.characteristic_speed_at_position(d_run, run_type)
        phases = (self.theta_phases[time_step] + self.phase_shifts) % 360
        relative_phases = (phases - self.phase_current) / self.phase_range_extent
        theta_distances = np.maximum(self.theta_time * characteristic_speed * self.multipliers[run_type]
                                     + self.adders[run_type], 0)
        represented_positions = self.positions[time_step] + (1 - 2*run_type) * theta_distances * relative_phases
        valid_positions = (represented_positions >= 0) & (represented_positions < self.track_length)
        represented_spatial_bins = np.round(np.ma.masked_array(represented_positions, mask=~valid_positions)
                                            / self.ds).astype(int)

        self.field_activations = np.zeros(self.num_cells)
        theta = np.cos(self.theta_phases[time_step] / 180 * np.pi)
        self.field_activations[valid_positions] = (self.fields[run_type,
                                                               represented_spatial_bins[valid_positions],
                                                               valid_positions]
                                                   * ((-theta + 1) / 2 * self.theta_modulation +
                                                      (1 - self.theta_modulation)))

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, p, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):

        model_spikes = cls(super_group_name, group_name, child_name, kwargs['LFP'], kwargs['Tracking'], p["num_cells"],
                           p["ds"], p["dt"], p["phase_range"], p["phase_current"], p["firing_rate_0"],
                           p["firing_rate_slope"], p["theta_modulation"], p['theta_time'], p['multiplicative_sigma'],
                           p['additive_sigma'], p['size_to_theta_d'], p['size_min'], p['shift_sigma'], None,
                           save_figures, figure_format, figures_path)
        model_spikes.generate_spikes()

        return model_spikes
