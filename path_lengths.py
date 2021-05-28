import sys
import json
import copy
import numpy as np
from numpy.polynomial import polynomial
from scipy import signal
from scipy.stats import linregress, theilslopes
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
from data_analysis.general import Base
from data_analysis.decoder import Decoder


class PathLengths(Base):
    """Calculate theta trajectory (path) lengths.

    Args:
        super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is defined
            as belonging to the super-group, it will be shared across sub-groups.
        group_name (string): Name of the low-level sub-group used for pickles and figures.
        child_name (string): Name of the instance used for pickles and figures.
        decoder (Decoder): Decoder's instance.
        spatial_extent (float): Size of the spatial window within which to assess theta trajectories (cm).
        fit_color (string): Matplotlib name for the color used for plotting the best fitting lines.

    Attributes:
        x_flat (np.array): Phases corresponding to every bin in the (position x phase) matrix stacked column-wise.
        y_flat (np.array): Positions corresponding to every bin in the (position x phase matrix) stacked column-wise.
        y (np.array): Positions corresponding to every bin in the (position x phase) matrix.
    """
    dependencies = (Decoder,)

    def __init__(self, super_group_name, group_name, child_name, decoder, spatial_extent=90, color_map='viridis',
                 fit_color='C1', save_figures=False, figure_format="png", figures_path="figures", pickle_results=True,
                 pickles_path="pickles"):

        super().__init__(super_group_name, group_name, child_name, save_figures, figure_format, figures_path,
                         pickle_results, pickles_path)

        self.decoder = decoder
        self.num_spatial_bins = int(round(spatial_extent/self.decoder.firing_fields.bin_size)) + 1
        self.spatial_extent = (self.num_spatial_bins - 1)*self.decoder.firing_fields.bin_size
        self.accepted_bins = None  # spatial bins included in place field or phase precession slope analyses
        self.color_map = color_map
        self.fit_color = fit_color

        self.phase_bins_per_cycle = self.decoder.phase_bins_per_cycle
        self.phase_extent = (self.decoder.phase_bin_size / 2,
                             (self.phase_bins_per_cycle - 1) * self.decoder.phase_step_size
                             + self.decoder.phase_bin_size / 2)
        self.phase_at_bins = np.linspace(self.phase_extent[0], self.phase_extent[1], self.phase_bins_per_cycle)

        # radon fit auxiliary variables
        self.y = None
        self.x_flat = None
        self.y_flat = None

        # variables for single cycle analyses
        self.sc_ok_indices = None
        self.sc_central_positions = None
        self.sc_speeds = None
        self.sc_decoded_positions_flat = None

    def calculate_accepted_bins(self, speed_groups, min_occupancy, min_spread):
        """Find out the spatial bins contained in place fields included in place field size or phase precession slopes
        due to their occupancy spread.
        """
        fields = self.decoder.firing_fields.screened_fields(include_incomplete=1)

        occupancy, smooth_occupancy, max_smooth_occupancies = \
            self.decoder.firing_fields.occupancy_by_speed(speed_groups)
        fields_spread = self.decoder.firing_fields.fields_spread(fields['run_types'], fields['bound_indices'],
                                                                 occupancy, min_occupancy)

        self.accepted_bins = np.full(occupancy.shape, False)
        for field_num, (run_type, bound_indices) in enumerate(zip(fields['run_types'], fields['bound_indices'])):
            within_field = slice(bound_indices[0], bound_indices[1] + 1)

            for speed_group_num in range(len(speed_groups)):
                if fields_spread[field_num, speed_group_num] > min_spread:
                    self.accepted_bins[run_type, speed_group_num, within_field] = True

    def accepted_bin(self, run_type, speed_group_num, position):
        return self.accepted_bins[run_type, speed_group_num, int(round(position/self.decoder.firing_fields.bin_size))]

    def averaged_path_lengths(self, names, speed_groups, margins, restricted_occupancies, min_cycles=5,
                              path_decoding="fit_max", hanning_width=14, radon_fit_params=None,
                              speed_groups_to_plot=(), cycles_fig_size=(12, 4)):

        if radon_fit_params is None:
            radon_fit_params = dict(num_slopes=20, slope_bounds=(-0.45, 0.45), num_intercepts=20,
                                    intercept_bounds=(-60, 60), d=20)

        fig, ax = plt.subplots(constrained_layout=True)
        for group_num, (name, speed_groups, margin_sizes, restricted_occupancy) in \
                enumerate(zip(names, [speed_groups for _ in range(len(margins))], margins, restricted_occupancies)):
            decoding_range = (margin_sizes[0] - self.decoder.tracking.d_runs_offset,
                              np.nanmax(self.decoder.tracking.d) - margin_sizes[1] - self.decoder.tracking.d_runs_offset)
            averaged_speeds, path_lengths = self.average_cycles(name, speed_groups, decoding_range,
                                                                restricted_occupancy, min_cycles, path_decoding,
                                                                hanning_width, radon_fit_params, speed_groups_to_plot,
                                                                fig_size=cycles_fig_size)

            ax.plot(averaged_speeds, path_lengths, '.-', color=f"C{group_num}", label=f"{name.lower()}")

            folder_name = f"averaged_cycles/{name.lower().replace(':', '').replace(' ', '_')}"
            self.maybe_pickle_results(averaged_speeds[np.newaxis], "speeds", subfolder=folder_name)
            self.maybe_pickle_results(path_lengths[np.newaxis], "path_lengths", subfolder=folder_name)

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Theta path length (cm)")
        ax.set_xlabel("Running speeds (cm/s)")
        ax.legend(fontsize="small")
        self.maybe_save_fig(fig, "theta_path_lengths", subfolder="averaged_cycles")

    def extract_cycle_probabilities(self, cycle_num, central_position):
        positions = (np.linspace(-self.spatial_extent / 2, self.spatial_extent / 2, self.num_spatial_bins)
                     + central_position)
        decoded_spatial_bins = np.round(positions / self.decoder.firing_fields.bin_size).astype(int)
        valid_spatial_bins = (decoded_spatial_bins >= 0) & (decoded_spatial_bins < self.decoder.num_spatial_bins)

        cycle_probabilities = np.full((self.num_spatial_bins, self.phase_bins_per_cycle), np.nan)
        cycle_probabilities[valid_spatial_bins] = \
            self.decoder.decoded_probabilities[decoded_spatial_bins[valid_spatial_bins],
                                               cycle_num * self.phase_bins_per_cycle:
                                               (cycle_num + 1) * self.phase_bins_per_cycle]
        return cycle_probabilities

    def plot_cycle_probabilities(self, probabilities, ax, v_max=None, x_tick_spacing=None, y_tick_spacing=None):
        c_map = copy.copy(plt.cm.get_cmap(self.color_map))
        c_map.set_bad(color='white')
        mat = ax.matshow(probabilities, origin="lower", aspect="auto", cmap=c_map, vmin=0, vmax=v_max,
                         extent=(self.phase_extent[0] - self.decoder.phase_step_size / 2,
                                 self.phase_extent[1] + self.decoder.phase_step_size / 2,
                                 - self.spatial_extent / 2 - self.decoder.firing_fields.bin_size / 2,
                                 + self.spatial_extent / 2 + self.decoder.firing_fields.bin_size / 2))
        ax.axhline(0, color="lightgray", linestyle="dashed")
        ax.xaxis.set_ticks_position("bottom")
        if x_tick_spacing is not None:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(x_tick_spacing))
        if y_tick_spacing is not None:
            ax.yaxis.set_major_locator(ticker.MultipleLocator(y_tick_spacing))
        return mat

    def extract_decoded_positions(self, cycle_probabilities, hanning_filter=None):
        if hanning_filter is not None:
            probabilities = np.empty(cycle_probabilities.shape)
            for phase_bin_num in range(self.phase_bins_per_cycle):
                probabilities[:, phase_bin_num] = signal.convolve(cycle_probabilities[:, phase_bin_num],
                                                                  hanning_filter, mode='same')
        else:
            probabilities = cycle_probabilities

        decoded_positions = np.full(cycle_probabilities.shape[1], np.nan)
        ok_cols = ~np.isnan(probabilities).all(axis=0)
        decoded_positions[ok_cols] = (np.nanargmax(probabilities[:, ok_cols], axis=0)
                                      * self.decoder.firing_fields.bin_size - self.spatial_extent / 2)
        return decoded_positions

    def average_cycles(self, title, speed_groups, decoding_range, restricted_occupancy, min_cycles,
                       path_decoding="fit_max", hanning_width=14, radon_fit_params=None, speed_groups_to_plot=(),
                       fig_size=(7, 1.8), coherent_color_code=True):

        averaged_probabilities = np.zeros((2, len(speed_groups), self.num_spatial_bins, self.phase_bins_per_cycle))
        averaged_speeds = np.zeros((2, len(speed_groups)))
        cycle_counts = np.zeros((2, len(speed_groups)))

        for cycle_num, (run_type, central_position, speed) in \
                enumerate(zip(self.decoder.run_types, self.decoder.central_positions, self.decoder.speeds)):
            if decoding_range[0] < central_position < decoding_range[1]:
                cycle_probabilities = np.nan_to_num(self.extract_cycle_probabilities(cycle_num, central_position))
                if run_type == 1:
                    cycle_probabilities = np.flip(cycle_probabilities, axis=0)
                for speed_group_num, speed_group in enumerate(speed_groups):
                    if speed_group[0] <= speed < speed_group[1]:
                        if (restricted_occupancy and self.accepted_bin(run_type, speed_group_num, central_position)
                                or not restricted_occupancy):
                            averaged_speeds[run_type, speed_group_num] += speed
                            averaged_probabilities[run_type, speed_group_num] += cycle_probabilities
                            cycle_counts[run_type, speed_group_num] += 1

        # combine run types, normalize probabilities & speeds, calculate decoded positions and/or theta path lengths
        phases = np.linspace(self.decoder.phase_bin_size / 2,
                             (self.phase_bins_per_cycle - 1) * self.decoder.phase_step_size
                             + self.decoder.phase_bin_size / 2, self.phase_bins_per_cycle)

        decoded_positions = np.full((len(speed_groups), self.phase_bins_per_cycle), np.nan)
        slopes = np.full(len(speed_groups), np.nan)
        intercepts = np.full(len(speed_groups), np.nan)
        path_lengths = np.full(len(speed_groups), np.nan)

        if path_decoding == "radon_fit" and self.y is None:
            self.set_radon_fit_auxiliary_variables()

        if path_decoding == "fit_smooth_max":
            hanning_width = ((hanning_width / self.decoder.firing_fields.bin_size) // 2) * 2 + 1
            hanning_filter = np.hanning(hanning_width)
            hanning_filter /= np.sum(hanning_filter)
        else:
            hanning_filter = None

        combined_probabilities = np.zeros((len(speed_groups), self.num_spatial_bins, self.phase_bins_per_cycle))
        combined_speeds = np.zeros(len(speed_groups))

        for speed_group_num in range(len(speed_groups)):
            cycle_count = np.sum(cycle_counts[:, speed_group_num])
            if cycle_count >= min_cycles:
                # normalize probabilities and speeds
                combined_probabilities[speed_group_num] += (averaged_probabilities[0, speed_group_num]
                                                            + averaged_probabilities[1, speed_group_num]) / cycle_count
                combined_speeds[speed_group_num] = np.sum(averaged_speeds[:, speed_group_num]) / cycle_count

                # fit theta path
                if path_decoding in ("fit_weighted_average", "fit_max", "fit_smooth_max"):
                    if path_decoding == "fit_weighted_average":
                        positions = np.linspace(-self.spatial_extent / 2, self.spatial_extent / 2,
                                                self.num_spatial_bins)
                        decoded_positions[speed_group_num] = \
                            np.dot(positions, combined_probabilities[speed_group_num] /
                                   np.sum(combined_probabilities[speed_group_num], axis=0))

                    elif path_decoding in ("fit_max", "fit_smooth_max"):
                        decoded_positions[speed_group_num] = self.extract_decoded_positions(
                            combined_probabilities[speed_group_num], hanning_filter)

                    slope, intercept = linregress(phases, decoded_positions[speed_group_num])[:2]

                elif path_decoding == "radon_fit":
                    p = self.radon_fit(combined_probabilities[speed_group_num], **radon_fit_params)
                    intercept = p[0]
                    slope = p[1]

                else:
                    sys.exit("Position decoding method not recognized.")

                slopes[speed_group_num] = slope
                intercepts[speed_group_num] = intercept
                path_lengths[speed_group_num] = slope*(self.phase_extent[1] - self.phase_extent[0])

        # plot averaged theta cycles
        if coherent_color_code:
            max_prob = combined_probabilities.max()
        else:
            max_prob = None
        fig, axes = plt.subplots(1, len(speed_groups_to_plot), sharex="col", sharey="col", figsize=fig_size,
                                 constrained_layout=True, squeeze=False)
        # fig.suptitle(title)
        col_num = 0
        for speed_group_num, speed_group in enumerate(speed_groups):
            if speed_group in speed_groups_to_plot:
                ax = axes[0, col_num]
                mat = self.plot_cycle_probabilities(combined_probabilities[speed_group_num], ax, v_max=max_prob,
                                                    x_tick_spacing=90, y_tick_spacing=15)

                ax.plot(phases, decoded_positions[speed_group_num], '.', color=self.fit_color)
                ax.plot(phases, slopes[speed_group_num] * phases + intercepts[speed_group_num], color=self.fit_color)

                ax.set_title(f"{speed_group} cm/s", fontsize='medium')
                ax.set_xlabel(r"$\theta$ phase"" (°)")
                if col_num == 0:
                    ax.set_ylabel("Position (cm)")
                elif coherent_color_code and col_num == len(speed_groups_to_plot) - 1:
                    bar = fig.colorbar(mat, ax=ax, aspect=60)
                    bar.ax.set_ylabel("Decoded probability")

                # for run_type in range(2):
                #     self.plot_cycle_probabilities(averaged_probabilities[run_type, speed_group_num],
                #                                   axes[1 + run_type, col_num], v_max=None,
                #                                   x_tick_spacing=90, y_tick_spacing=15)

                col_num += 1

        fig_name = f"averaged_cycles_{title.lower().replace(':', '').replace(' ', '_')}"
        self.maybe_save_fig(fig, fig_name, subfolder="averaged_cycles")

        return combined_speeds, path_lengths

    def single_cycles(self, min_peak_prob, min_phase_coverage, min_phase_extent, radon_fit_params,
                      max_cycles_to_plot, from_run_types, cycles_per_figure, sort_by_speed=True, flip=True,
                      cycles_fig_size=(11, 6), plot_summary=False, summary_fig_size=(10, 7),
                      save_decoded_positions=True):

        self.sc_ok_indices = [[], []]
        self.sc_central_positions = [[], []]
        self.sc_speeds = [[], []]
        sc_decoded_positions = [[], []]
        path_lengths = [[], []]
        max_probabilities = [[], []]
        poly_fits = [[], []]

        if radon_fit_params is None:
            radon_fit_params = dict(num_slopes=20, slope_bounds=(-0.45, 0.45), num_intercepts=20,
                                    intercept_bounds=(-60, 60), d=20)

        phase_extent = self.phase_extent[1] - self.phase_extent[0]
        if min_phase_coverage > phase_extent or min_phase_extent > phase_extent:
            sys.exit(f"min_phase_coverage or min_phase_extent can't be smaller than {phase_extent}.")
        min_phase_bins = int(round(min_phase_coverage/phase_extent*(self.phase_bins_per_cycle - 1)))
        min_phase_extent_bins = int(round(min_phase_extent/phase_extent*(self.phase_bins_per_cycle - 1)))

        if self.y is None:
            self.set_radon_fit_auxiliary_variables()

        for cycle_num, (run_type, central_position, speed) in \
                enumerate(zip(self.decoder.run_types, self.decoder.central_positions, self.decoder.speeds)):
            probabilities = self.extract_cycle_probabilities(cycle_num, central_position)

            # skip invalid theta cycles
            if np.isnan(probabilities).all():
                continue
            phase_bin_ok = np.nanmax(np.nan_to_num(probabilities), axis=0) > min_peak_prob
            if np.sum(phase_bin_ok) < min_phase_bins:
                continue
            if (len(phase_bin_ok) - np.argwhere(phase_bin_ok[::-1])[0, 0] - np.argwhere(phase_bin_ok)[0, 0]
                    < min_phase_extent_bins):
                continue

            # fit
            p = self.radon_fit(probabilities, **radon_fit_params)

            self.sc_ok_indices[run_type].append(cycle_num)
            self.sc_central_positions[run_type].append(central_position)
            self.sc_speeds[run_type].append(speed)
            path_lengths[run_type].append(p[1] * (self.phase_extent[1] - self.phase_extent[0]))
            max_probabilities[run_type].append(np.nanmax(probabilities))
            poly_fits[run_type].append(p)

            if save_decoded_positions:
                sc_decoded_positions[run_type].append(self.extract_decoded_positions(probabilities))

        if save_decoded_positions:
            self.sc_decoded_positions_flat = np.array(
                [(1 - 2 * run_type) * z for run_type, x in enumerate(sc_decoded_positions) for y in x for z in y])
            self.maybe_pickle_results(self.sc_decoded_positions_flat, "decoded_positions", subfolder="single_cycles")

        if max_cycles_to_plot:
            self.plot_single_cycles(max_probabilities, poly_fits, max_cycles_to_plot, from_run_types, cycles_per_figure,
                                    sort_by_speed, flip, cycles_fig_size)

        speeds = [np.array(array) for array in self.sc_speeds]
        central_positions = [np.array(array) for array in self.sc_central_positions]
        distances_from_start = [central_positions[0], self.decoder.tracking.d_runs_span - central_positions[1]]
        normalized_pos = [distances_from_start[0] / self.decoder.tracking.d_runs_span,
                          distances_from_start[1] / self.decoder.tracking.d_runs_span]
        characteristic_speeds = []
        for run_type in range(2):
            run_type_characteristic_speeds = []
            for sc_central_position in self.sc_central_positions[run_type]:
                c_speed = self.decoder.tracking.characteristic_speed_at_position(sc_central_position, run_type)
                run_type_characteristic_speeds.append(c_speed)
            characteristic_speeds.append(np.array(run_type_characteristic_speeds))
        path_lengths = [np.array(array) for array in path_lengths]

        self.maybe_pickle_results(speeds, "speeds", subfolder="single_cycles")
        self.maybe_pickle_results(distances_from_start, "distances_from_start", subfolder="single_cycles")
        self.maybe_pickle_results(normalized_pos, "normalized_pos", subfolder="single_cycles")
        self.maybe_pickle_results([path_lengths[0], -path_lengths[1]], "path_lengths", subfolder="single_cycles")
        self.maybe_pickle_results(characteristic_speeds, "characteristic_speeds", subfolder="single_cycles")

        if plot_summary:
            self.single_cycles_summary(path_lengths, fig_size=summary_fig_size)

    def plot_single_cycles(self, max_probabilities, poly_fits, max_cycles_to_plot, from_run_types, cycles_per_figure,
                           sort_by_speed=True, flip=True, fig_size=(11, 6), coherent_color_code=False):

        # select random subset of cycles (discarding the few that have negative speeds)
        ok_cycle_indices = np.array([y for x in [self.sc_ok_indices[run_type] for run_type in from_run_types] for y in x])
        speeds = np.array([y for x in [self.sc_speeds[run_type] for run_type in from_run_types] for y in x])
        max_probabilities = np.array([y for x in [max_probabilities[run_type] for run_type in from_run_types] for y in x])
        poly_fits = np.array([y for x in [poly_fits[run_type] for run_type in from_run_types] for y in x])

        cycles_to_plot = min(max_cycles_to_plot, len(ok_cycle_indices))
        selected_indices = np.random.choice(len(ok_cycle_indices), cycles_to_plot, replace=False)
        if sort_by_speed:
            selected_indices = selected_indices[np.argsort(speeds[selected_indices])]

        selected_cycle_indices = ok_cycle_indices[selected_indices]
        selected_speeds = speeds[selected_indices]
        selected_max_probabilities = max_probabilities[selected_indices]
        selected_poly_fits = poly_fits[selected_indices]

        # create figures
        figs = []
        axes = []
        total_cycles_per_figure = cycles_per_figure[0] * cycles_per_figure[1]
        num_figures = np.ceil(max_cycles_to_plot / total_cycles_per_figure).astype(int)
        for _ in range(num_figures):
            fig, ax = plt.subplots(cycles_per_figure[0], cycles_per_figure[1], sharey="row", figsize=fig_size)
            # ax[-1, 0].set_xlabel(r"$\theta$ Phase (°)")
            # ax[-1, 0].set_ylabel("Position (cm)")
            figs.append(fig)
            axes.append(ax)

        # plot cycles
        for fig_num, fig in enumerate(figs):
            start = total_cycles_per_figure * fig_num
            end = min(cycles_to_plot, total_cycles_per_figure * (fig_num + 1))

            if coherent_color_code:
                fig.subplots_adjust(left=0.15)
                ax_pos = axes[fig_num][-2, 0].get_position()
                cax_pos = [ax_pos.x0 - 0.08, ax_pos.y0, 0.01, ax_pos.height]
                c_ax = fig.add_axes(cax_pos)
                max_probability = np.max(selected_max_probabilities[start:end])
                norm = Normalize(vmin=0., vmax=max_probability)
                fig.colorbar(cm.ScalarMappable(norm=norm, cmap=self.color_map), cax=c_ax, aspect=10)
                c_ax.yaxis.set_ticks_position('left')
            else:
                max_probability = None

            for fig_cycle_num, (cycle_index, speed, poly_fit) in \
                    enumerate(zip(selected_cycle_indices[start:end], selected_speeds[start:end],
                                  selected_poly_fits[start:end])):
                central_position = self.decoder.central_positions[cycle_index]
                probabilities = self.extract_cycle_probabilities(cycle_index, central_position)
                if flip and self.decoder.run_types[cycle_index]:
                    probabilities = np.flip(probabilities, axis=0)
                    poly_fit *= -1
                row_num = fig_cycle_num // cycles_per_figure[1]
                col_num = fig_cycle_num % cycles_per_figure[1]
                ax = axes[fig_num][row_num, col_num]
                self.plot_cycle_probabilities(probabilities, ax, v_max=max_probability, x_tick_spacing=90,
                                              y_tick_spacing=25)
                ax.plot(self.phase_at_bins, polynomial.polyval(self.phase_at_bins, poly_fit), color=self.fit_color)
                annotation = ax.annotate(f"{speed:.1f}", (0.95, 0.10), xycoords='axes fraction', color="#D65F5F",
                                         fontsize='small', horizontalalignment='right')
                # annotation.set_path_effects([path_effects.Stroke(linewidth=0.5, foreground='black'),
                #                              path_effects.Normal()])
                ax.tick_params(axis='x', rotation=90, length=1.5)
                ax.tick_params(axis='y', length=1.5)

                if row_num != cycles_per_figure[0] - 1:
                    ax.set_xticklabels([])

        # save figures
        for figure_num, figure in enumerate(figs):
            figure.tight_layout(h_pad=0.5, w_pad=0.5)
            self.maybe_save_fig(figure, f"batch_{figure_num}", subfolder="single_cycles")

    def single_cycles_summary(self, path_lengths, fig_size=(10, 7)):
        """Plot single cycle path lengths vs instantaneous speed, characteristic speed at the cycles' locations or
        position along the track.
        """
        fig, ax = plt.subplots(2, 3, sharey="all", sharex="col", constrained_layout=True, figsize=fig_size)
        ax[1, 0].set_xlabel("Running speed (cm/s)")
        ax[1, 1].set_xlabel("Mean running speed (cm/s)")
        ax[1, 2].set_xlabel("Position (cm)")

        for run_type, run_type_name in enumerate(self.decoder.tracking.run_type_names):
            run_type_lengths = (1 - 2 * run_type) * path_lengths[run_type]
            if len(run_type_lengths) > 2:
                # slopes vs instantaneous speed
                run_type_speeds = np.array(self.sc_speeds[run_type])
                ax[run_type, 0].axhline(0, linestyle="dotted", color="C7")
                ax[run_type, 0].plot(run_type_speeds, run_type_lengths, '.', color=f"C{run_type}", markersize=1,
                                     label=f"{run_type_name} cycles")
                ax[run_type, 0].set_ylabel("Decoded theta trajectory\nlength (cm)")

                slope, intercept, r_value, p_value, stderr = linregress(run_type_speeds, run_type_lengths)
                lin_speeds = np.array([run_type_speeds.min(), run_type_speeds.max()])
                ax[run_type, 0].plot(lin_speeds, lin_speeds*slope + intercept,
                                     label=f"p={p_value:.2e}, r={r_value:.2f}", color='k', linewidth=1)
                ax[run_type, 0].legend(loc="lower right", fontsize="x-small")

                # slopes vs median speed
                ax[run_type, 1].axhline(0, linestyle="dotted", color="C7")
                run_type_characteristic_speeds = []
                for sc_central_position in self.sc_central_positions[run_type]:
                    c_speed = self.decoder.tracking.characteristic_speed_at_position(sc_central_position, run_type)
                    run_type_characteristic_speeds.append(c_speed)
                run_type_characteristic_speeds = np.array(run_type_characteristic_speeds)
                ax[run_type, 1].plot(run_type_characteristic_speeds, run_type_lengths, '.', color=f"C{run_type}",
                                     markersize=1)
                slope, intercept, r_value, p_value, stderr = \
                    linregress(run_type_characteristic_speeds[~np.isnan(run_type_characteristic_speeds)],
                               run_type_lengths[~np.isnan(run_type_characteristic_speeds)])
                lin_speeds = np.array([np.nanmin(run_type_characteristic_speeds),
                                       np.nanmax(run_type_characteristic_speeds)])
                ax[run_type, 1].plot(lin_speeds, lin_speeds*slope + intercept,
                                     label=f"p={p_value:.2e}, r={r_value:.2f}", color='k', linewidth=1)
                ax[run_type, 1].legend(loc="lower right", fontsize="x-small")

                # slopes vs position
                ax[run_type, 2].axhline(0, linestyle="dotted", color="C7")
                ax[run_type, 2].plot(self.sc_central_positions[run_type], run_type_lengths, '.', color=f"C{run_type}",
                                     markersize=1)

        self.maybe_save_fig(fig, "path_lengths", subfolder="single_cycles")

    def radon_fit(self, probabilities, num_slopes, slope_bounds, num_intercepts, intercept_bounds, d):
        """Finds the rectangular area with highest value of the sum probability and then performs a linear weighted fit.

        Args:
            probabilities (np.array): Matrix of decoded probabilities.
            num_slopes (int): Number of slopes to try out.
            slope_bounds (tuple(float)): Lower and upper bounds for the slopes (cm/deg).
            num_intercepts (int): Number of intercepts to try out per slope.
            intercept_bounds (tuple(float)): Lower and upper bounds for the intercepts (cm).
            d (float): Orthogonal distance from the fitting line which determines the half width of the window (cm).
        """
        best_prob_sum = 0
        best_slope = 0
        best_intercept = 0

        num_probabilities = np.nan_to_num(probabilities)

        for slope in np.linspace(slope_bounds[0], slope_bounds[1], num_slopes):
            d_y = d*np.sqrt(1 + slope**2)
            for intercept in np.linspace(intercept_bounds[0], intercept_bounds[1], num_intercepts):
                lower_bound = slope*self.phase_at_bins + intercept - d_y
                upper_bound = slope*self.phase_at_bins + intercept + d_y
                within_bounds = (self.y >= lower_bound) & (self.y <= upper_bound)
                if within_bounds.any():
                    prob_sum = np.sum(num_probabilities[within_bounds])
                    if prob_sum > best_prob_sum:
                        best_prob_sum = prob_sum
                        best_slope = slope
                        best_intercept = intercept

        d_y = d * np.sqrt(1 + best_slope ** 2)
        lower_bound = best_slope * self.phase_at_bins + best_intercept - d_y
        upper_bound = best_slope * self.phase_at_bins + best_intercept + d_y
        within_bounds = (self.y >= lower_bound) & (self.y <= upper_bound)
        roi_probabilities = np.where(within_bounds, num_probabilities, 0)
        p = polynomial.polyfit(self.x_flat, self.y_flat, 1, w=roi_probabilities.flatten())

        return p

    def set_radon_fit_auxiliary_variables(self):
        self.y = np.linspace(-self.spatial_extent/2*np.ones(self.phase_bins_per_cycle),
                             self.spatial_extent/2*np.ones(self.phase_bins_per_cycle), self.num_spatial_bins)
        self.x_flat = (np.ones((self.num_spatial_bins, self.phase_bins_per_cycle)) * self.phase_at_bins).flatten()
        self.y_flat = self.y.flatten()

    def correlate_decoded_and_predicted(self, predicted, plot=False, y_sigma=1.2, fig_name="decoded vs predicted",
                                        subfolder="single_cycles", marker_size=0.5):
        if len(self.sc_decoded_positions_flat):
            not_nan = ~np.isnan(self.sc_decoded_positions_flat) & ~np.isnan(predicted)
            slope, intercept, r, p, e = linregress(predicted[not_nan], self.sc_decoded_positions_flat[not_nan])

            # reg = RANSACRegressor().fit(decoded[not_nan][np.newaxis].T, predicted[not_nan])
            # ransac_slope = reg.estimator_.coef_[0]
            # ransac_r = reg.score(decoded[not_nan][np.newaxis].T, predicted[not_nan])

            if plot:
                fig, ax = plt.subplots()
                ax.plot(predicted[not_nan], self.sc_decoded_positions_flat[not_nan]
                        + np.random.normal(scale=y_sigma, size=np.sum(not_nan)), '.',
                        markersize=marker_size, alpha=0.5)
                x = np.array((np.nanmin(predicted), np.nanmax(predicted)))

                ax.plot(x, x*slope + intercept, label=f"slope = {slope:.2f}; R = {r:.2f}")
                # ax.plot(x, reg.predict(x[np.newaxis].T), label=f'RANSAC slope = {ransac_slope:.2f}: R = {ransac_r:.2f}')
                ax.legend()
                ax.set_xlabel("Predicted position")
                ax.set_ylabel("Decoded position")
                self.maybe_save_fig(fig, fig_name, subfolder=subfolder)

            return slope, intercept, r
        else:
            return np.nan, np.nan, np.nan

    def displacement_compensation(self, run_type, cycle_index, central_position):
        cycle_real_positions = (self.decoder.real_positions[cycle_index * self.phase_bins_per_cycle:
                                                            (cycle_index + 1) * self.phase_bins_per_cycle]
                                - central_position)
        if run_type == 1:
            cycle_real_positions *= -1
        return cycle_real_positions

    def decoded_vs_average_speed_predictions(self, theta_time, phase_current=180, plot=False,
                                             displacement_compensation=True):
        predicted_positions = np.empty(0)
        for run_type in range(2):
            for cycle_index, central_position in zip(self.sc_ok_indices[run_type], self.sc_central_positions[run_type]):
                speed = self.decoder.tracking.characteristic_speed_at_position(central_position, run_type)
                cycle_predicted_positions = (self.phase_at_bins - phase_current) / 360 * theta_time * speed
                if displacement_compensation:
                    cycle_predicted_positions += self.displacement_compensation(run_type, cycle_index, central_position)
                predicted_positions = np.append(predicted_positions, cycle_predicted_positions)

        fig_name = f"decoded vs predicted - theta_time = {theta_time:.2f}" \
                   f"{', displacement compensated' if displacement_compensation else ''}"

        self.maybe_pickle_results(predicted_positions, f"predicted_positions_{theta_time:.2f}",
                                  subfolder='single_cycles/speed')
        slope, intercept, r = self.correlate_decoded_and_predicted(predicted_positions, plot=plot, fig_name=fig_name,
                                                                   subfolder='single_cycles/speed')
        return slope, intercept, r

    def RK4_predictor(self, cycle_real_positions, cycle_predicted_positions, central_position, phase_bins, theta_time,
                      run_type):
        previous_phase = 180
        previous_position = central_position
        for phase in phase_bins:
            previous_phase_index = np.nonzero(self.phase_at_bins == previous_phase)
            phase_index = np.nonzero(self.phase_at_bins == phase)
            real_increment = cycle_real_positions[phase_index] - cycle_real_positions[previous_phase_index]
            if np.isnan(real_increment).any():
                return
            k1 = self.decoder.tracking.characteristic_speed_at_position(previous_position, run_type)
            if np.isnan(k1):
                return
            halfway_position_1 = previous_position + real_increment/2 + (phase - previous_phase) / 720 * theta_time * k1
            k2 = self.decoder.tracking.characteristic_speed_at_position(halfway_position_1, run_type)
            if np.isnan(k2):
                return
            halfway_position_2 = previous_position + real_increment/2 + (phase - previous_phase) / 720 * theta_time * k2
            k3 = self.decoder.tracking.characteristic_speed_at_position(halfway_position_2, run_type)
            if np.isnan(k3):
                return
            end_position = previous_position + real_increment + (phase - previous_phase) / 360 * theta_time * k3
            k4 = self.decoder.tracking.characteristic_speed_at_position(end_position, run_type)
            if np.isnan(k4):
                return

            speed = (k1 + 2*k2 + 2*k3 + k4)/6

            new_position = previous_position + real_increment + (phase - previous_phase) / 360 * theta_time * speed
            cycle_predicted_positions[self.phase_at_bins == phase] = new_position - central_position
            previous_phase = phase
            previous_position = new_position

    def decoded_vs_average_time_predictions(self, theta_time, plot=False, displacement_compensation=False):
        predicted_positions = np.empty(0)
        central_phase_bin = int(round((self.phase_bins_per_cycle - 1) / 2))
        if self.phase_bins_per_cycle % 2 == 0:
            print("This is meant to work with an odd number of phase bins per cycle!")
        for run_type in range(2):
            for cycle_index, central_position in zip(self.sc_ok_indices[run_type], self.sc_central_positions[run_type]):
                cycle_predicted_positions = np.full(self.phase_bins_per_cycle, np.nan)
                cycle_predicted_positions[central_phase_bin] = 0
                if displacement_compensation:
                    cycle_real_positions = self.displacement_compensation(run_type, cycle_index, central_position)
                else:
                    cycle_real_positions = np.zeros(self.phase_bins_per_cycle)
                self.RK4_predictor(cycle_real_positions, cycle_predicted_positions, central_position,
                                   self.phase_at_bins[central_phase_bin+1:], theta_time, run_type)
                self.RK4_predictor(cycle_real_positions, cycle_predicted_positions, central_position,
                                   self.phase_at_bins[:central_phase_bin][::-1], theta_time, run_type)
                predicted_positions = np.append(predicted_positions, cycle_predicted_positions)

        fig_name = f"decoded vs predicted - theta_time = {theta_time:.2f}" \
                   f"{', displacement compensated' if displacement_compensation else ''}"

        self.maybe_pickle_results(predicted_positions, f"predicted_positions_{theta_time:.2f}",
                                  subfolder='single_cycles/time')
        slope, intercept, r = self.correlate_decoded_and_predicted(predicted_positions, plot=plot, fig_name=fig_name,
                                                                   subfolder=f'single_cycles/time')
        return slope, intercept, r

    def optimize_theta_times(self, bounds, num_points, model_type, displacement_compensation=True, plot=True,
                             pickle_best=True):
        theta_times = np.linspace(bounds[0], bounds[1], num_points)
        slopes = []
        rs = []
        for theta_time in theta_times:
            if model_type == 'time':
                slope, intercept, r = \
                    self.decoded_vs_average_time_predictions(theta_time,
                                                             displacement_compensation=displacement_compensation)
            elif model_type == 'speed':
                slope, intercept, r = \
                    self.decoded_vs_average_speed_predictions(theta_time,
                                                              displacement_compensation=displacement_compensation)
            else:
                sys.exit("'model_type' should be 'time' or 'speed'")
            slopes.append(slope)
            rs.append(r)

        subfolder = f"single_cycles/{model_type}"

        if plot:
            fig, ax = plt.subplots()
            ax.plot(theta_times, slopes)
            ax.set_xlabel("Theta time (s)")
            ax.set_ylabel("Slope of decoded position vs predicted position")
            fig_name = f"slopes vs theta_time{', displacement compensated' if displacement_compensation else ''}"
            self.maybe_save_fig(fig, fig_name, subfolder=subfolder)

        best_index = np.argmin(np.abs(np.array(slopes) - 1))

        if pickle_best:
            self.maybe_pickle_results(theta_times[best_index], "theta_time", subfolder=subfolder)
            self.maybe_pickle_results(rs[best_index], "R", subfolder=subfolder)

        return theta_times[best_index]

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):
        return cls(super_group_name, group_name, child_name, kwargs['Decoder'], parameters_dict['spatial_extent'],
                   save_figures=save_figures, figure_format=figure_format, figures_path=figures_path,
                   pickle_results=pickle_results, pickles_path=pickles_path)


