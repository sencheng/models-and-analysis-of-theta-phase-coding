import os
import glob
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import linregress, wilcoxon, kendalltau
import warnings
import matplotlib.pyplot as plt
import matplotlib.colors as colors_lib
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from data_analysis.analyze.config import speed_groups, general_parameters, pickles_path
from data_analysis.analyze.batch_config import sessions, rats, figures_path, group_names
from data_analysis.initializer import initialize
from data_analysis.tracking import Tracking
from misc.likelihood_ratio_test import hierarchical_lrt


# colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#77BEDB"]
violin_color = "#4878CF"
plt.rcParams.update({'figure.constrained_layout.use': True})
plt.rcParams.update({'savefig.dpi': 100})

average_speeds = np.mean(speed_groups, axis=1)


def load(session, group_name, rel_path):
    """Load a pickle.
    """
    if group_name is not None:
        path = f"{pickles_path}/{session}/{group_name}/{rel_path}"
    else:
        path = f"{pickles_path}/{session}/{rel_path}"
    with open(path, 'rb') as f:
        return pickle.load(f)


def ok_bounds(values, inter_quartile_factor=3):
    """Bounds of accepted values defined in terms of the inter-quartile range.
    """
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    span = inter_quartile_factor * (q3 - q1)
    return q1 - span, q3 + span


def load_pooled(group_name, rel_paths, remove_outliers=False, inter_quartile_factor=3):
    """Load speed-pooled data into values_container.
    """
    values_container = [[[] for _ in rats] for _ in rel_paths]
    sessions_container = [[] for _ in rats]

    for session in sessions:
        rat_index = rats.index(session.split('.')[0])
        for var_num, rel_path in enumerate(rel_paths):
            rat_values = load(session, group_name, rel_path)
            for values in rat_values:
                if (~np.isnan(values)).any():
                    values_container[var_num][rat_index].append(values)
                    if var_num == 0:
                        sessions_container[rat_index].append(session)

    if remove_outliers:
        for var_num in range(len(rel_paths)):
            for rat_index in range(len(rats)):
                lower_bound, upper_bound = ok_bounds(values_container[var_num][rat_index], inter_quartile_factor)
                with np.errstate(invalid='ignore'):
                    out_of_bounds = ((values_container[var_num][rat_index] < lower_bound) |
                                     (values_container[var_num][rat_index] > upper_bound))
                values_container[var_num][rat_index] = np.where(out_of_bounds, np.nan,
                                                                values_container[var_num][rat_index])

    return values_container, sessions_container


def nan_regress(x, y, only_slope=True):
    """Perform linear regression ignoring nans.
    """
    not_nan = ~np.isnan(y) & ~np.isnan(x)
    if np.sum(not_nan):
        fit = linregress(x[not_nan], y[not_nan])
        if only_slope:
            return fit[0]
        else:
            return fit
    else:
        return np.nan


def summary_pooled(rats_values, axes, axes_summary, averaged_speeds, name, mean_line_width=1.5, slopes_x_label='',
                   p_pos=0, min_points=5):
    rats_speed_slopes = [[] for _ in rats]
    max_num_values = len(average_speeds)

    mean_rats_values = []

    # plot shaded area with std
    all_values = np.vstack(rats_values)
    mean = np.nanmean(all_values, axis=0)
    std = np.nanstd(all_values, axis=0)
    axes_summary[1].fill_between(average_speeds, mean-std/2, mean+std/2, color='C7', alpha=0.2)

    for rat_index in range(len(rats)):
        rat_speed_slope_weights = []
        rat_values = np.array(rats_values[rat_index])
        mean_rat_values = []
        for column in rat_values.T:
            if np.sum(~np.isnan(column)) >= min_points:
                mean_rat_values.append(np.nanmean(column))
            else:
                mean_rat_values.append(np.nan)
        mean_rats_values.append(mean_rat_values)

        flat_rat_values = rat_values.flatten()
        flat_speeds = np.concatenate([averaged_speeds for _ in range(rat_values.shape[0])])
        grand_regress = nan_regress(flat_speeds, flat_rat_values, only_slope=False)

        # calculate field speed slopes and their weights
        for values in rat_values:
            not_nan = ~np.isnan(values)
            not_nan_sum = np.sum(not_nan)
            if not_nan_sum > 1:
                rats_speed_slopes[rat_index].append(linregress(averaged_speeds[not_nan], values[not_nan])[0])
                rat_speed_slope_weights.append(not_nan_sum/max_num_values)

        axes[rat_index].plot(average_speeds, average_speeds * grand_regress.slope + grand_regress.intercept, color='k',
                             linestyle='dashed', label=f'fit')
        axes[rat_index].annotate(f"p={grand_regress.pvalue:.1e}", (0.6, 0.85), xycoords="axes fraction",
                                 fontsize="x-small")
        print(f"{rats[rat_index]}:\nWald test: N = {np.sum(~np.isnan(flat_rat_values))}, R = {grand_regress.rvalue},"
              f"p = {grand_regress.pvalue:.2e}")

        not_nan = ~np.isnan(flat_speeds) & ~np.isnan(flat_rat_values)
        tau, p_tau = kendalltau(flat_speeds[not_nan], flat_rat_values[not_nan])
        print(f"Kendal Tau test: Tau = {tau}, p = {p_tau}")

        summary_general(rat_index, averaged_speeds, mean_rat_values, rats_speed_slopes, rat_speed_slope_weights,
                        grand_regress, axes, axes_summary, name, mean_line_width=mean_line_width,
                        slopes_x_label=slopes_x_label, p_pos=p_pos)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grand_mean = np.nanmean(mean_rats_values, axis=0)

    axes_summary[1].plot(average_speeds, grand_mean, 'k', linewidth=mean_line_width)


def summary_general(rat_index, mean_x, mean_y, slopes, slope_weights, grand_regress, axes, axes_summary, name,
                    plot_slopes=True, remove_outliers=True, inter_quartile_factor=3, min_marker_size=1,
                    max_marker_size=5, mean_line_width=1.5, wilcoxon_test=True, slopes_x_label='', p_pos=0):
    axes[rat_index].plot(mean_x, mean_y, color='k', linewidth=mean_line_width, label='mean')

    axes_summary[1].plot(mean_x, mean_y, '.-', label=rats[rat_index], color=colors[rat_index])
    axes_summary[1].set_ylabel(name)
    axes_summary[0].set_xlabel(slopes_x_label)

    # mean and single-field place field size vs speed slopes
    if plot_slopes:
        if remove_outliers:
            lower_bound, upper_bound = ok_bounds(slopes[rat_index], inter_quartile_factor)
            slopes_ok = (slopes[rat_index] >= lower_bound) & (slopes[rat_index] <= upper_bound)
            slopes[rat_index] = np.array(slopes[rat_index])[slopes_ok]
            slope_weights = np.array(slope_weights)[slopes_ok]

        y_bottom = sum([len(slopes[smaller_rat_index]) for smaller_rat_index in range(rat_index)])
        y_top = y_bottom + len(slopes[rat_index])

        axes_summary[0].plot(grand_regress.slope * np.ones(2), (y_bottom, y_top), color='C7', label="regress all")
        # mean_rat_value_slopes = nan_regress(mean_x, mean_y)
        # axes_summary[0].plot(mean_rat_value_slopes * np.ones(2), (y_bottom, y_top), color='C7', label="f(mean(x))")

        if rat_index == len(rats) - 1:
            axes_summary[0].set_ylim([0, y_top])
        else:
            axes_summary[0].axhline(y_top, color='C7', zorder=0)
        axes_summary[0].axvline(0, color='C7', linestyle='dotted')
        axes_summary[0].plot(np.average(slopes[rat_index], weights=slope_weights) * np.ones(2), (y_bottom, y_top),
                             color='k', label="mean")

        sorted_indices = np.argsort(slopes[rat_index])
        marker_sizes = min_marker_size + (max_marker_size-min_marker_size)*np.array(slope_weights)[sorted_indices]
        axes_summary[0].scatter(np.array(slopes[rat_index])[sorted_indices], range(y_bottom, y_top),
                                color=colors[rat_index], s=marker_sizes, label=rats[rat_index], linewidths=0.0)

        if wilcoxon_test:
            if len(slopes[rat_index]) >= 10:
                statistic, p_value = wilcoxon(slopes[rat_index])
                axes_summary[0].annotate(f"p = {p_value:.2f}", (p_pos, (y_bottom + y_top)/2), xycoords='data',
                                         fontsize='small', verticalalignment='center')
                print(f"Wilcoxon test: N = {len(slopes[rat_index])}, p = {p_value:.2e}")

    else:
        axes_summary[0].set_axis_off()


def finish(fig, axes, name, group_name, summary=False, x_labels=None, y_label="Place field #", close_figure=1):
    # title = fig.suptitle(name)
    # bbox_extra_artists = [title]
    bbox_extra_artists = []

    if x_labels is None:
        x_labels = ["Running speed (cm/s)"]

    for row_num, x_label in zip(range(axes.shape[0] - 1, -1, -1), x_labels[::-1]):
        if not summary:
            for col_num in range(axes.shape[-1]):
                axes[row_num, col_num].set_xlabel(x_label)
        else:
            axes[row_num, 1].set_xlabel(x_label)

    if summary:
        handles, labels = axes[-1][0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        legend = axes[-1][0].legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, -0.25),
                                    fontsize='x-small', ncol=4)
        legend.get_frame().set_linewidth(0.6)
        bbox_extra_artists.append(legend)

        for row_num in range(axes.shape[0]):
            axes[row_num, 0].set_ylabel(y_label)
            # axes[row_num, 0].axhline(0, color='C7', linewidth=1)
            x_lim = axes[row_num, 0].get_xlim()
            lim = np.max(np.abs(x_lim))
            axes[row_num, 0].set_xlim((-lim, lim))

            # axes[row_num, 1].yaxis.set_major_formatter(FormatStrFormatter('%i'))
            axes[row_num, 1].yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        for col_num in range(axes.shape[-1]):
            axes[0, col_num].set_title(rats[col_num])

            handles, full_labels = axes[-1, col_num].get_legend_handles_labels()
            labels = []
            for label in full_labels:
                if '.' in label:
                    labels.append(label.split('.')[1])
                else:
                    labels.append(label)
            by_label = dict(zip(labels, handles))
            legend = axes[-1, col_num].legend(by_label.values(), by_label.keys(), loc='upper left', fontsize='x-small',
                                              ncol=2, bbox_to_anchor=(0, -0.45), borderaxespad=0)
            bbox_extra_artists.append(legend)

    for row_num in range(axes.shape[0]):
        for col_num in range(axes.shape[-1]):
            axes[row_num, col_num].spines['right'].set_visible(False)
            axes[row_num, col_num].spines['top'].set_visible(False)

    fig.align_ylabels()
    group_name_path = f"{group_name}/" if group_name is not None else ''
    fig.savefig(f"{figures_path}/{group_name_path}{name}", bbox_extra_artists=bbox_extra_artists,
                bbox_inches='tight')
    if close_figure:
        plt.close(fig)


def pooled_violins(title, group_name, names, rats_pooled_values, pairwise_increments=True, bottom_percentile=None,
                   top_percentile=None, increments_bottom_percentile=None, increments_top_percentile=None, bw=None,
                   ax=None, increments_ax=None, plot_y_labels=True, increments_y_labels=None,
                   fig_size=None, close_figure=1, non_overlapping_increments=False):

    if ax is None:
        own_figure = True
        fig, axes = plt.subplots(len(names), 1 + pairwise_increments, sharex='col', squeeze=False, figsize=fig_size)
        ax = axes[:, 0]
        if pairwise_increments:
            increments_ax = axes[:, 1]
        else:
            increments_ax = None
    else:
        own_figure = False

    for row_num, name in enumerate(names):
        all_values = rats_pooled_values[row_num][0]
        for rat_index in range(1, len(rats) - 1):
            all_values = np.vstack((all_values, rats_pooled_values[row_num][rat_index]))
        clean_values = []
        x = []

        for group_num, values_group in enumerate(all_values.T):
            clean_values_group = values_group[~np.isnan(values_group)]
            if len(clean_values_group):
                clean_values.append(clean_values_group)
                x.append(average_speeds[group_num])

        parts = ax[row_num].violinplot(clean_values, x, showmeans=False, showextrema=False, bw_method=bw,
                                       widths=0.5 * (average_speeds[1] - average_speeds[0]))
        for pc in parts['bodies']:
            pc.set_alpha(0.5)
            pc.set_facecolor(violin_color)
        ax[row_num].plot(average_speeds, np.nanmean(all_values, axis=0), '.-', color='k')

        if bottom_percentile is not None and top_percentile is not None:
            ax[row_num].set_ylim([0.9 * np.nanpercentile(all_values, bottom_percentile),
                                  np.nanpercentile(all_values, top_percentile)])
        if plot_y_labels:
            ax[row_num].set_ylabel(name)

        if pairwise_increments:
            if non_overlapping_increments:
                # find non-overlapping speed groups
                indices = [0]
                upper_bound = speed_groups[0][-1]
                for group_num, speed_group in enumerate(speed_groups[1:]):
                    if speed_group[0] >= upper_bound:
                        indices.append(group_num + 1)
                        upper_bound = speed_group[-1]
            else:
                indices = list(range(len(speed_groups)))

            increments = np.diff(all_values[:, indices])
            increments_x_values = []
            clean_increments = []

            for increments_group_num, increments_group in enumerate(increments.T):
                clean_increments_group = increments_group[~np.isnan(increments_group)]
                if len(clean_increments_group):
                    clean_increments.append(clean_increments_group)
                    increments_x_values.append((average_speeds[increments_group_num]
                                                + average_speeds[increments_group_num + 1]) / 2)

            parts = increments_ax[row_num].violinplot(clean_increments, positions=increments_x_values, showmeans=False,
                                                      showextrema=False, bw_method=bw,
                                                      widths=0.5 * (average_speeds[1] - average_speeds[0]))
            for pc in parts['bodies']:
                pc.set_alpha(0.5)
                pc.set_facecolor(violin_color)
            increments_ax[row_num].plot(increments_x_values,
                                        [np.mean(increments_group) for increments_group in clean_increments],
                                        '.', color='k')
            increments_ax[row_num].axhline(0, linestyle='dotted', color='C7')
            if plot_y_labels:
                if increments_y_labels is not None:
                    y_label = increments_y_labels[row_num]
                else:
                    y_label = f"Within-field\n"rf"$\Delta$ {name.lower()}"
                increments_ax[row_num].set_ylabel(y_label)
            if increments_bottom_percentile is not None and increments_top_percentile is not None:
                increments_ax[row_num].set_ylim([np.nanpercentile(increments, increments_bottom_percentile),
                                                 np.nanpercentile(increments, increments_top_percentile)])

    if own_figure:
        fig.savefig(f"{figures_path}/{group_name}/{title} - Violins")
        if close_figure:
            plt.close(fig)


def plot_pooled(title, names, paths, remove_outliers=False, inter_quartile_factor=3, alpha=0.6,
                pairwise_increments=True, bottom_percentile=None, top_percentile=None, increments_bottom_percentile=None,
                increments_top_percentile=None, violin_ax=None, violin_increments_ax=None,
                violin_increments_y_labels=None, fig_size=None, fig_summary_size=None, fig_violins_size=None,
                close_figures=1, slopes_x_label='', p_pos=0, summary_y_lims=None, min_points=5, vertical_layout=False):

    for group_num, group_name in enumerate(group_names):
        num_rows = len(names)
        fig, ax = plt.subplots(num_rows, len(rats), sharey='row', figsize=fig_size, squeeze=False)

        if vertical_layout:
            fig_summary, ax_s = plt.subplots(2, num_rows, figsize=fig_summary_size, squeeze=False)
            ax_summary = ax_s.T
        else:
            fig_summary, ax_summary = plt.subplots(num_rows, 2, figsize=fig_summary_size, squeeze=False)

        if summary_y_lims is not None:
            for axis in ax_summary[:, -1]:
                axis.set_ylim(summary_y_lims)

        rats_pooled_values, rats_sessions = load_pooled(group_name, paths, remove_outliers, inter_quartile_factor)

        for row_num, name in enumerate(names):
            for rat_index in range(len(rats)):
                for values, session in zip(rats_pooled_values[row_num][rat_index], rats_sessions[rat_index]):
                    session_num = [s.split('.')[0] for s in
                                   sessions[:sessions.index(session)]].count(session.split('.')[0])
                    ax[row_num, rat_index].plot(average_speeds, values, color=colors[session_num % len(colors)],
                                                alpha=alpha, label=session)
            ax[row_num, 0].set_ylabel(name)

            summary_pooled(rats_pooled_values[row_num], ax[row_num], ax_summary[row_num], average_speeds, name,
                           slopes_x_label=slopes_x_label, p_pos=p_pos, min_points=min_points)

        finish(fig, ax, f"{title} - All", group_name, close_figure=close_figures)
        finish(fig_summary, ax_summary, f"{title} - Summary", group_name, summary=True, close_figure=close_figures)

        pooled_violins(title, group_name, names, rats_pooled_values, pairwise_increments, bottom_percentile,
                       top_percentile, increments_bottom_percentile, increments_top_percentile,
                       ax=violin_ax[:, group_num] if violin_ax is not None else None,
                       increments_ax=violin_increments_ax[:, group_num] if violin_increments_ax is not None else None,
                       plot_y_labels=group_num == 0, increments_y_labels=violin_increments_y_labels
                       if violin_increments_y_labels is not None else None, fig_size=fig_violins_size,
                       close_figure=close_figures)


def clean_scatter(values, inter_quartile_factor):
    lower_bound, upper_bound = ok_bounds(np.concatenate(values), inter_quartile_factor)
    for group_num, group_y in enumerate(values):
        field_y_array = np.array(group_y)
        outlier = (field_y_array < lower_bound) | (field_y_array > upper_bound)
        values[group_num] = np.where(outlier, np.nan, values[group_num])


def load_scatter(group_name, paths_x, paths_y, remove_outliers=False, inter_quartile_factor=3, inverse=False):
    x_containers = [[[] for _ in rats] for _ in paths_x]
    y_containers = [[[] for _ in rats] for _ in paths_y]
    sessions_container = [[] for _ in rats]

    for session in sessions:
        rat_index = rats.index(session.split('.')[0])
        for x_num, (path_x, x_container) in enumerate(zip(paths_x, x_containers)):
            x = load(session, group_name, path_x)
            for y_num, (path_y, y_container) in enumerate(zip(paths_y, y_containers)):
                y = load(session, group_name, path_y)
                for x_group, y_group in zip(x, y):
                    if len(x_group):
                        if y_num == 0:
                            x_container[rat_index].append(np.array(x_group))
                        if x_num == 0 and not inverse:
                            y_container[rat_index].append(np.array(y_group))
                        elif x_num == 0 and inverse:
                            y_container[rat_index].append(1/np.array(y_group))
                        if x_num == y_num == 0:
                            sessions_container[rat_index].append(session)

    if remove_outliers:
        for y_num in range(len(paths_y)):
            for rat_index in range(len(rats)):
                clean_scatter(y_containers[y_num][rat_index], inter_quartile_factor)

    return x_containers, y_containers, sessions_container


def moving_average(xs, ys, window_size, window_stride, window_min_points, min_x=None, return_std=False):
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    if min_x is None:
        min_x = np.nanmin(x) - window_size / 2
    max_x = np.nanmax(x) + window_size / 2
    start_xs = [min_x]
    while start_xs[-1] + window_size < max_x:
        start_xs.append(start_xs[-1] + window_stride)
    end_xs = np.array(start_xs) + window_size
    mean_x = (start_xs + end_xs) / 2

    mean_y = []
    std = []
    for start_x, end_x in zip(start_xs, end_xs):
        y_window = y[(start_x < x) & (x < end_x)]
        if len(y_window) >= window_min_points:
            mean_y.append(np.nanmean(y_window))
            std.append(np.nanstd(y_window))
        else:
            mean_y.append(np.nan)
            std.append(np.nan)

    if return_std:
        return mean_x, np.array(mean_y), np.array(std)
    else:
        return mean_x, np.array(mean_y)


def moving_density(xs, ys, window_size, window_stride, min_x=None):
    x = np.concatenate(xs)
    y = np.concatenate(ys)

    if min_x is None:
        min_x = np.nanmin(x)
    max_x = np.nanmax(x)
    start_xs = [min_x]
    while start_xs[-1] + window_size < max_x:
        start_xs.append(start_xs[-1] + window_stride)
    end_xs = np.array(start_xs) + window_size
    mean_x = (start_xs + end_xs) / 2

    dens = []
    for start_x, end_x in zip(start_xs, end_xs):
        y_window = y[(start_x < x) & (x < end_x)]
        dens.append(np.sum(~np.isnan(y_window)))

    return mean_x, np.array(dens)/window_size


def summary_scatter(x, y, window_size, window_stride, window_min_points, axes, axes_summary, name, plot_slopes,
                    mean_line_width=1.5, min_x=0, p_pos=0, slopes_x_lim=None, slopes_x_label='', shade=False,
                    annotation_xy=(0.6, 0.15)):
    rats_slopes = [[] for _ in rats]
    max_count = max([max([len(group_x) for group_x in x[rat_index]]) for rat_index in range(len(rats))])
    mean_rats_y = []
    longest_mean_x = []

    # shaded area with std
    if shade:
        mean_x, mean_y, std = moving_average(np.concatenate(x), np.concatenate(y), window_size, window_stride,
                                             window_min_points, min_x=min_x - window_size/2, return_std=True)
        axes_summary[1].fill_between(mean_x, mean_y - std/2, mean_y + std/2, color="C7", alpha=0.2)

    for rat_index in range(len(rats)):
        # calculate group slopes and weights
        slope_weights = []
        for x_group, y_group in zip(x[rat_index], y[rat_index]):
            if len(x_group) > 1:
                slope_weights.append(len(x_group)/max_count)
                rats_slopes[rat_index].append(nan_regress(x_group, y_group))

        mean_x, mean_y = moving_average(x[rat_index], y[rat_index], window_size, window_stride, window_min_points,
                                        min_x=min_x - window_size/2)
        mean_rats_y.append(mean_y)
        if len(mean_x) > len(longest_mean_x):
            longest_mean_x = mean_x

        if plot_slopes:
            grand_regress = nan_regress(np.concatenate(x[rat_index]), np.concatenate(y[rat_index]), only_slope=False)
            regression_line_x = np.array((np.nanmin(np.concatenate(x[rat_index])), np.nanmax(np.concatenate(x[rat_index]))))
            # axes[rat_index].plot(regression_line_x, regression_line_x * grand_regress.slope + grand_regress.intercept,
            #                      color='k', linestyle='dashed', label=f'fit')
            # axes[rat_index].annotate(f"p={grand_regress.pvalue:.1e}", (0.6, 0.15), xycoords="axes fraction",
            #                          fontsize="x-small")
            n = np.sum(~np.isnan(np.concatenate(x[rat_index])) & ~np.isnan(np.concatenate(y[rat_index])))
            print(f"{rats[rat_index]}:\nWald test: N = {n}, p = {grand_regress.pvalue:.2e}")

            all_x = np.concatenate(x[rat_index])
            all_y = np.concatenate(y[rat_index])
            not_nan = ~np.isnan(all_x) & ~np.isnan(all_y)
            tau, p_tau = kendalltau(all_x[not_nan], all_y[not_nan])
            print(f"Kendal Tau test: Tau = {tau}, p = {p_tau}")
            axes[rat_index].annotate(f"p={p_tau:.1e}", annotation_xy, xycoords="axes fraction",
                                     fontsize="x-small")
        else:
            grand_regress = np.nan

        summary_general(rat_index, mean_x, mean_y, rats_slopes, slope_weights, grand_regress, axes, axes_summary, name,
                        plot_slopes, p_pos=p_pos, slopes_x_label=slopes_x_label)
        if slopes_x_lim is not None:
            axes_summary[0].set_xlim(slopes_x_lim)

    max_mean_length = max([len(mean_y) for mean_y in mean_rats_y])
    padded_rats_means = [np.pad(mean_y, (0, max_mean_length - len(mean_y)), constant_values=np.nan)
                         for mean_y in mean_rats_y]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        grand_mean = np.nanmean(padded_rats_means, axis=0)

    axes_summary[1].plot(longest_mean_x, grand_mean, 'k', linewidth=mean_line_width)


def remove_nans(all_xs, all_y):
    not_nans = ~np.isnan(all_y)
    for all_x in all_xs:
        not_nans &= ~np.isnan(all_x)
    all_xs = np.array([all_x[not_nans] for all_x in all_xs])
    return all_xs, all_y[not_nans]


def scatter_violins(all_x, all_y, name_x, name_y, ax, bw=0.5):
    # remove nans
    not_nan = ~(np.isnan(all_x) | np.isnan(all_y))
    all_x = all_x[not_nan]
    all_y = all_y[not_nan]

    # divide points into speed groups
    grouped_ys = [[] for _ in range(len(speed_groups))]

    for x, y in zip(all_x, all_y):
        for speed_group_num, speed_group in enumerate(speed_groups):
            if speed_group[0] <= x < speed_group[1]:
                grouped_ys[speed_group_num].append(y)

    # plot violin
    ax.violinplot(grouped_ys, average_speeds, showmeans=False, showextrema=False, bw_method=bw,
                  widths=0.5 * (average_speeds[1] - average_speeds[0]))
    ax.plot(average_speeds, [np.mean(group_ys) for group_ys in grouped_ys], '.-', color='k')
    ax.set_ylabel(name_y)
    ax.set_xlabel(name_x)


def sci_notation(num, decimal_digits=1, precision=None, exponent=None):
    """
    Returns a string representation of the scientific
    notation of the given number formatted for use with
    LaTeX or Mathtext, with specified number of significant
    decimal digits and precision (number of decimal digits
    to show). The exponent to be used can also be specified
    explicitly.
    """
    if num == 0:
        return ''
    else:
        if exponent is None:
            exponent = int(np.floor(np.log10(abs(num))))
        coeff = round(num / float(10**exponent), decimal_digits)
        if precision is None:
            precision = decimal_digits

        return r"${0:.{2}f}\cdot10^{{{1:d}}}$".format(coeff, exponent, precision)


def two_d_hist(all_x, all_y, x_label, y_label, fig_name, x_min, x_max, num_x_bins, y_min, y_max, num_y_bins,
               num_x_ticks=5, logarithm=True, line_width=1.5, close_figures=True, fig_size=(5.5/2.54, 5.56/2.54),
               min_points=5):
    x_bin_size = (x_max - x_min) / (num_x_bins - 1)
    y_bin_size = (y_max - y_min) / (num_y_bins - 1)
    hist = np.zeros((num_y_bins, num_x_bins))
    for x, y in zip(all_x, all_y):
        if x_min <= x <= x_max and y_min <= y <= y_max:
            hist[int(round((y - y_min) / y_bin_size)), int(round((x - x_min) / x_bin_size))] += 1

    mean = np.sum(np.linspace(y_min, y_max, num_y_bins)[np.newaxis].T * hist, axis=0) / np.sum(hist, axis=0)
    mean[np.sum(hist, axis=0) < min_points] = np.nan

    if logarithm:
        hist += 1
        norm = colors_lib.LogNorm(vmin=1)
    else:
        norm = colors_lib.Normalize(vmin=0)

    fig, ax = plt.subplots(figsize=fig_size)
    heatmap = ax.matshow(hist, origin='lower', norm=norm, aspect='auto',
                         extent=(x_min - x_bin_size/2, x_max + x_bin_size /2,
                                 y_min - y_bin_size/2, y_max + y_bin_size/2))
    bar = plt.colorbar(heatmap, ax=ax, aspect=60, orientation='horizontal')
    bar.ax.set_xlabel("Count + 1")
    ax.axhline(0, linestyle='dashed', color='white')
    ax.plot(np.linspace(x_min, x_max, num_x_bins), mean, 'k', linewidth=line_width)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label.replace('\n', ' '))
    ax.xaxis.set_ticks_position("bottom")
    ax.set_xticks(np.linspace(x_min, x_max, num_x_ticks))

    if close_figures:
        plt.close(fig)
    fig.savefig(fig_name, bbox_inches='tight')


def plot_scatter(title, names_x, paths_x, names_y, paths_y, window_sizes, window_strides, window_min_points, plot_slopes,
                 remove_outliers=False, inter_quartile_factor=3, alpha=0.6, summary=True, all_together=None,
                 all_together_means=None, all_together_dens=None, all_together_hists=None, hist_dicts=None,
                 plot_violins=(0,), analysis_of_variance=False, little_names_x=None,
                 fig_size=None, fig_summary_size=None, fig_all_size=None, fig_violins_size=None, close_figures=1,
                 p_pos=0, slopes_x_lim=None, slopes_x_label='', hyperbolic_fit=None, annotation_xy=(0.08, 0.1),
                 mean_line_width=1.5, shade=False, inverse=False, y_lim=None):
    for group_name in group_names:
        num_rows = max(len(names_x), len(names_y))
        fig, ax = plt.subplots(num_rows, len(rats), sharey='row', figsize=fig_size, squeeze=False)

        rats_xs, rats_ys, rats_sessions = load_scatter(group_name, paths_x, paths_y, remove_outliers,
                                                       inter_quartile_factor, inverse=inverse)

        for rat_index in range(len(rats)):
            row_num = 0
            for y_num, name_y in enumerate(names_y):
                for x_num, name_x in enumerate(names_x):
                    for x_group, y_group, session in zip(rats_xs[x_num][rat_index], rats_ys[y_num][rat_index],
                                                         rats_sessions[rat_index]):
                        session_num = [s.split('.')[0] for s in
                                       sessions[:sessions.index(session)]].count(session.split('.')[0])
                        ax[row_num, rat_index].plot(x_group, y_group, '.', color=colors[session_num % len(colors)],
                                                    alpha=alpha, label=session)
                    ax[row_num, 0].set_ylabel(name_y)
                    row_num += 1

        if all_together is not None:
            fig_all, ax_all = plt.subplots(sum(all_together), 1, sharey='row', squeeze=False, figsize=fig_all_size)
            if y_lim is not None:
                for axis_all in ax_all:
                    axis_all[0].set_ylim(y_lim)

            if sum(plot_violins):
                fig_violins, ax_violins = plt.subplots(sum(plot_violins), 1, sharex='col', squeeze=False,
                                                       figsize=fig_violins_size)

            def hyperbola(x_point, theta_time):
                return -360/(x_point * theta_time)

            row_num = 0
            combo_num = 0
            violin_num = 0
            for y_num, name_y in enumerate(names_y):
                all_y_arrays = []
                for rat_index in range(len(rats)):
                    for y_group in rats_ys[y_num][rat_index]:
                        all_y_arrays.append(y_group)
                all_y = np.concatenate(all_y_arrays)

                all_xs = []
                for x_num, name_x in enumerate(names_x):
                    if all_together[combo_num]:
                        all_x_arrays = []
                        for rat_index in range(len(rats)):
                            for group_num, x_group in enumerate(rats_xs[x_num][rat_index]):
                                all_x_arrays.append(x_group)
                                ax_all[row_num, 0].plot(x_group, rats_ys[y_num][rat_index][group_num], '.',
                                                        color=colors[rat_index], alpha=alpha, label=rats[rat_index])
                                ax_all[row_num, 0].set_ylabel(names_y[y_num])
                                ax_all[row_num, 0].set_xlabel(names_x[x_num])
                                ax_all[row_num, 0].spines['top'].set_visible(False)
                                ax_all[row_num, 0].spines['right'].set_visible(False)

                        all_x = np.concatenate(all_x_arrays)
                        all_xs.append(all_x)

                        if all_together_hists is not None and all_together_hists[combo_num]:
                            fig_name = f"{figures_path}/{group_name}/{title} - 2D histogram  {combo_num}"
                            two_d_hist(all_x, all_y, name_x, name_y, fig_name, **hist_dicts[combo_num],
                                       close_figures=close_figures)

                        if all_together_means is not None and all_together_means[combo_num]:
                            mean_x, mean_y = moving_average(all_x_arrays, all_y_arrays, window_sizes[combo_num],
                                                            window_strides[combo_num],
                                                            window_min_points, min_x=0 - window_sizes[combo_num] / 2)
                            ax_all[row_num, 0].plot(mean_x, mean_y, 'k', linewidth=mean_line_width)

                        if all_together_dens is not None and all_together_dens[combo_num]:
                            mean_x, dens = moving_density(all_x_arrays, all_y_arrays, window_sizes[combo_num],
                                                          window_strides[combo_num])
                            ax_dens = ax_all[row_num, 0].twinx()
                            ax_dens.plot(mean_x, dens)
                            ax_dens.set_ylabel("Density")

                        if plot_slopes[combo_num]:
                            if hyperbolic_fit is not None and hyperbolic_fit[combo_num]:
                                tau = - 360 * nan_regress(all_x, 1 / all_y)
                                x = np.linspace(np.nanmin(all_x), np.nanmax(all_x), 100)
                                y = hyperbola(x, tau)
                                ax_all[row_num, 0].plot(x, y, 'k', linestyle='dashed')
                                annotation = r"m = $-\frac{360}{"+str(round(tau, 2))+"\cdot v}$"
                                ax_all[row_num, 0].annotate(annotation, xy=(0.1, 0.1), xycoords='axes fraction',
                                                            horizontalalignment='left')

                            fit = nan_regress(all_x, all_y, only_slope=False)
                            x = np.array((np.nanmin(all_x), np.nanmax(all_x)))
                            y = x * fit[0] + fit[1]
                            ax_all[row_num, 0].plot(x, y, 'k')
                            # p_string = sci_notation(fit[3])
                            ax_all[row_num, 0].annotate(rf"$R^2$ = {fit[2]**2:.2f}""\n",
                                                        xy=annotation_xy, xycoords='axes fraction')
                            print(f"ALL: \nWald Test: N = {np.sum(~np.isnan(all_x) & ~np.isnan(all_y))}, "
                                  f"p = {fit[3]:.2e}")

                            not_nan = ~np.isnan(all_y) & ~np.isnan(all_x)
                            tau, p_tau = kendalltau(all_x[not_nan], all_y[not_nan])
                            print(f"ALL: Kendal Tau test: Tau = {tau}, p = {p_tau}")

                        if sum(plot_violins) and plot_violins[combo_num]:
                            scatter_violins(all_x, all_y, name_x, name_y, ax_violins[violin_num][0])
                            violin_num += 1

                        row_num += 1
                    combo_num += 1

                if analysis_of_variance:
                    if little_names_x is None:
                        little_names_x = [name_x for combo_num, name_x in enumerate(names_x)
                                          if all_together[combo_num] and plot_slopes[combo_num]]
                    all_xs, all_y = remove_nans(all_xs, all_y)
                    hierarchical_lrt(all_xs[np.array(plot_slopes).astype(bool)].T, all_y[np.newaxis].T, little_names_x)

            if sum(plot_violins):
                fig_violins.savefig(f"{figures_path}/{group_name}/{title} - Violins", bbox_inches='tight')
                if close_figures:
                    plt.close(fig_violins)

            handles, labels = ax_all[-1, -1].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax_all[-1, -1].legend(by_label.values(), by_label.keys(), loc='upper center', fontsize='small',
                                  bbox_to_anchor=(0.5, -0.3), ncol=3)
            fig_all.savefig(f"{figures_path}/{group_name}/{title} - Pooled", bbox_inches='tight')
            if close_figures:
                plt.close(fig_all)

        if summary:
            fig_summary, ax_summary = plt.subplots(num_rows, 2, figsize=fig_summary_size, squeeze=False)
            row_num = 0
            for x in rats_xs:

                all_x = []
                for rat_index in range(len(rats)):
                    for x_group in x[rat_index]:
                        all_x.append(x_group)
                all_x = np.concatenate(all_x)
                min_x = np.nanmin(all_x)

                for y, y_name in zip(rats_ys, names_y):
                    summary_scatter(x, y, window_sizes[row_num], window_strides[row_num], window_min_points, ax[row_num],
                                    ax_summary[row_num], y_name, plot_slopes[row_num], min_x=min_x, p_pos=p_pos,
                                    slopes_x_lim=slopes_x_lim, slopes_x_label=slopes_x_label, shade=shade,
                                    annotation_xy=annotation_xy)
                    row_num += 1
            finish(fig_summary, ax_summary, f"{title} - Summary", group_name, summary=True, x_labels=names_x,
                   close_figure=close_figures)

        fig.tight_layout(h_pad=3)
        finish(fig, ax, f"{title} - All", group_name, x_labels=names_x, close_figure=close_figures)


def cycle_lengths_vs_field_size(path_field_distances, path_field_sizes, path_trajectory_distances,
                                path_trajectory_lengths, window_size, window_stride, window_min_points,
                                remove_outliers=True, inter_quartile_factor=3, alpha=0.6, fig_size=None,
                                close_figure=1):
    for group_name in group_names:
        field_distances, field_sizes, rats_sessions = load_scatter(group_name, [path_field_distances],
                                                                   [path_field_sizes])

        trajectory_distances, trajectory_slopes, rats_sessions = load_scatter(group_name, [path_trajectory_distances],
                                                                              [path_trajectory_lengths])

        fig, ax = plt.subplots(1, len(rats), sharey='row', figsize=fig_size, squeeze=False)
        ax[0, 0].set_ylabel("Theta trajectory\nslope (cm/deg)")

        for rat_index in range(len(rats)):
            if remove_outliers:
                clean_scatter(field_sizes[0][rat_index], inter_quartile_factor)
                clean_scatter(trajectory_slopes[0][rat_index], inter_quartile_factor)

            mean_distances, mean_sizes = moving_average(field_distances[0][rat_index], field_sizes[0][rat_index],
                                                        window_size, window_stride, window_min_points)

            min_distance = mean_distances[0]
            ds = mean_distances[1] - mean_distances[0]

            all_sizes = []
            all_slopes = []
            for group_trajectory_distances, group_trajectory_slopes, group_session in \
                    zip(trajectory_distances[0][rat_index], trajectory_slopes[0][rat_index], rats_sessions[rat_index]):
                sizes = []
                slopes = []
                for trajectory_distance, trajectory_slope in zip(group_trajectory_distances, group_trajectory_slopes):
                    i_previous = int((trajectory_distance - min_distance) / ds)
                    i_next = i_previous + 1
                    remainder = (trajectory_distance - min_distance) % ds
                    if 0 <= i_previous and i_next < len(mean_sizes):
                        sizes.append(mean_sizes[i_previous] * (1-remainder) + mean_sizes[i_next] * remainder)
                        slopes.append(trajectory_slope)

                session_num = [s.split('.')[0] for s in
                               sessions[:sessions.index(group_session)]].count(group_session.split('.')[0])

                ax[0, rat_index].plot(sizes, slopes, '.', color=colors[session_num % len(colors)], alpha=alpha,
                                      label=group_session)

                all_sizes.append(sizes)
                all_slopes.append(slopes)

            all_sizes = np.concatenate(all_sizes)
            all_slopes = np.concatenate(all_slopes)
            x = np.array((np.nanmin(all_sizes), np.nanmax(all_sizes)))
            fit = nan_regress(all_sizes, all_slopes, only_slope=False)
            ax[0, rat_index].plot(x, x*fit[0] + fit[1], 'k')
            ax[0, rat_index].annotate(f"s = {fit[1]:.1e} + {fit[0]:.1e} v\nr = {fit[2]:.2f}\np = {fit[3]:.1e}",
                                      xy=(0.08, 0.8), xycoords='axes fraction', fontsize='x-small')

        finish(fig, ax, "Single cycle theta trajectories vs mean place field size", group_name,
               x_labels=["Mean place field size\nat trajectory location(cm)"], close_figure=close_figure)


def speed_histograms(plot_colorbars=True, fig_size=None, close_figure=1):
    p = general_parameters['Tracking']
    fig, ax = plt.subplots(1, len(rats), sharey='row', figsize=fig_size)
    ax[0].set_ylabel("Speed (cm/s)")

    # find largest values for displacement and speed
    max_ds = [[] for _ in rats]
    max_speeds = [[] for _ in rats]

    for session in sessions:
        rat_index = rats.index(session.split('.')[0])
        tracking = initialize((Tracking,), session, None)['Tracking']
        max_speeds[rat_index].append(np.nanmax(np.abs(tracking.speed_1D[tracking.run_type != -1])))
        max_ds[rat_index].append(tracking.d_runs_span)
        del tracking

    max_speed = np.max(np.concatenate(max_speeds)) * 1.05

    # calculate and plot speed histograms
    for rat_index, rat in enumerate(rats):
        speed_bin_size = p['speed_bin_size']
        num_speed_bins = int(round(max_speed / speed_bin_size)) + 1

        max_d = max(max_ds[rat_index])
        spatial_bin_size = p['spatial_bin_size']
        num_spatial_bins = int(round(max_d / spatial_bin_size)) + 1

        speeds_vs_position = np.zeros((num_speed_bins, num_spatial_bins))

        for session in sessions:
            if rat in session:
                tracking = initialize((Tracking,), session, None)['Tracking']
                session_speeds_vs_position = tracking.calculate_histograms(num_spatial_bins, spatial_bin_size,
                                                                           tracking.speed_1D, num_speed_bins,
                                                                           speed_bin_size)
                speeds_vs_position += session_speeds_vs_position[0]
                speeds_vs_position += np.fliplr(session_speeds_vs_position[1])

        Tracking.plot_histogram(axes=ax[rat_index], y_vs_position=speeds_vs_position, min_d=0,
                                num_spatial_bins=num_spatial_bins, spatial_bin_size=spatial_bin_size,
                                threshold_y=0, num_y_bins=num_speed_bins, y_bin_size=speed_bin_size,
                                max_v=np.max(speeds_vs_position), plot_colorbar=plot_colorbars,
                                x_label="Distance from the\nstart of run (cm)")

        ax[rat_index].set_title(rat)
    fig.savefig(f"{figures_path}/speed_histograms", bbox_inches='tight')
    if close_figure:
        plt.close(fig)


def characteristic_speeds(fig_name, paths, spatial_bin_size, alpha=0.6, mean_line_width=1.5, fig_size=None,
                          fig_summary_size=None, close_figures=1):
    fig, ax = plt.subplots(1, len(rats), sharey='row', squeeze=False, figsize=fig_size)
    ax[0, 0].set_ylabel("Characteristic\nrunning speed (cm/s)")
    rats_speeds, rats_sessions = load_pooled(None, [paths])
    rats_means = []
    for rat_index in range(len(rats)):
        for run_type_speeds, session in zip(rats_speeds[0][rat_index], rats_sessions[rat_index]):
            session_num = [s.split('.')[0] for s in
                           sessions[:sessions.index(session)]].count(session.split('.')[0])
            ax[0, rat_index].plot(np.arange(len(run_type_speeds))*spatial_bin_size, run_type_speeds,
                                  color=colors[session_num % len(colors)], alpha=alpha, label=session)
        max_bins = max([len(speeds) for speeds in rats_speeds[0][rat_index]])
        balanced_array = []
        for speeds in rats_speeds[0][rat_index]:
            balanced_array.append(np.append(speeds, np.full(max_bins - len(speeds), np.nan)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            rat_means = np.nanmean(balanced_array, axis=0)
        rats_means.append(rat_means)
        ax[0, rat_index].plot(np.arange(max_bins)*spatial_bin_size, rat_means, 'k', linewidth=mean_line_width,
                              label='mean')
    finish(fig, ax, fig_name, None, x_labels=["Distance from the\nstart of the run (cm)"], close_figure=close_figures)

    # summary
    fig_summary, ax_summary = plt.subplots(figsize=fig_summary_size)
    ax_summary.set_ylabel("Characteristic\nrunning speed (cm/s)")
    ax_summary.set_xlabel("Normalized run distance")
    ax_summary.spines['top'].set_visible(False)
    ax_summary.spines['right'].set_visible(False)

    max_bins = max([len(rat_means) for rat_means in rats_means])
    upsampled_speeds = []
    upsampled_mean_speeds = []
    for rat_speeds, rat_means in zip(rats_speeds[0], rats_means):
        interpolation = interp1d(np.linspace(0, 1, len(rat_means)), rat_means)
        upsampled_mean_speeds.append(interpolation(np.linspace(0, 1, max_bins)))
        for speeds in rat_speeds:
            interpolation = interp1d(np.linspace(0, 1, len(speeds)), speeds)
            upsampled_speeds.append(interpolation(np.linspace(0, 1, max_bins)))

    mean = np.nanmean(upsampled_speeds, axis=0)
    std = np.nanstd(upsampled_speeds, axis=0)
    ax_summary.fill_between(np.linspace(0, 1, max_bins), mean - std/2, mean + std/2, color='C7', alpha=0.2)
    for rat_index, (rat_means, rat) in enumerate(zip(rats_means, rats)):
        rat_x = np.linspace(0, 1, len(rat_means))
        ax_summary.plot(rat_x, rat_means, label=rat, alpha=alpha, color=colors[rat_index])

    ax_summary.plot(np.linspace(0, 1, max_bins), np.nanmean(upsampled_mean_speeds, axis=0), 'k',
                    linewidth=mean_line_width, label='mean')

    legend = ax_summary.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=4, fontsize='x-small')
    fig_summary.savefig(f"{figures_path}/{fig_name} - Summary", bbox_extra_artists=[legend], bbox_inches='tight')
    if close_figures:
        plt.close(fig)
        plt.close(fig_summary)


def pooled_and_scatter_summary(fig_name, pooled_names, pooled_paths, scatter_path_x, scatter_names_y, scatter_paths_y,
                               remove_outliers=False, inter_quartile_factor=3, window_size=10, window_stride=5,
                               window_min_points=25, alpha=1, thick_line_width=1.5, origin=(0, 0), fig_size=(3, 4),
                               close_figure=1):
    for group_name in group_names:
        fig, ax = plt.subplots(len(pooled_names) + len(scatter_names_y), sharex='col', sharey='col', figsize=fig_size)
        ax[-1].set_xlabel("Running speed (cm/s)")

        # plot scatter data
        rats_xs, rats_ys, rats_sessions = load_scatter(group_name, [scatter_path_x], scatter_paths_y, remove_outliers,
                                                       inter_quartile_factor)

        for row_num, scatter_name_y in enumerate(scatter_names_y):
            ax[row_num].set_ylabel(scatter_name_y)

            mean_rats_x = []
            mean_rats_y = []

            mean_x, mean_y, std = moving_average(np.concatenate(rats_xs[0]), np.concatenate(rats_ys[row_num]),
                                                 window_size, window_stride, window_min_points, min_x=0, return_std=True)
            ax[row_num].fill_between(mean_x, mean_y-std/2, mean_y+std/2, color='C7', alpha=0.2)

            longest_mean_x = []
            for rat_index, rat in enumerate(rats):
                mean_x, mean_y = moving_average(rats_xs[0][rat_index], rats_ys[row_num][rat_index], window_size,
                                                window_stride, window_min_points, min_x=0)
                if len(mean_x) > len(longest_mean_x):
                    longest_mean_x = mean_x
                mean_rats_x.append(mean_x)
                mean_rats_y.append(mean_y)
                ax[row_num].plot(mean_x, mean_y, '.-', alpha=alpha, label=rat, color=colors[rat_index])

            max_mean_length = max([len(mean_y) for mean_y in mean_rats_y])
            padded_rats_means = [np.pad(mean_y, (0, max_mean_length - len(mean_y)), constant_values=np.nan) for mean_y
                                 in mean_rats_y]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                grand_mean = np.nanmean(padded_rats_means, axis=0)
            ax[row_num].plot(longest_mean_x, grand_mean, 'k', linewidth=thick_line_width)

            ax[-1].set_xlim(left=origin[0], right=None)
            ax[-1].set_ylim(bottom=origin[1], top=None)

        # plot pooled data
        rats_pooled_values, rats_sessions = load_pooled(group_name, pooled_paths, remove_outliers,
                                                        inter_quartile_factor)
        mean_rats_values = []
        for pooled_num, pooled_name in enumerate(pooled_names):
            row_num = len(scatter_names_y) + pooled_num
            ax[row_num].set_ylabel(pooled_name)
            for rat_index, rat in enumerate(rats):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_rat_values = np.nanmean(rats_pooled_values[pooled_num][rat_index], axis=0)
                mean_rats_values.append(mean_rat_values)
                ax[row_num].plot(average_speeds, mean_rat_values, '.-', label=rat, alpha=alpha, color=colors[rat_index])

            mean_y = np.nanmean(np.vstack(rats_pooled_values[pooled_num]), axis=0)
            std = np.nanstd(np.vstack(rats_pooled_values[pooled_num]), axis=0)
            ax[row_num].fill_between(average_speeds, mean_y - std/2, mean_y + std/2, color='C7', alpha=0.2)

            ax[row_num].plot(average_speeds, np.nanmean(mean_rats_values, axis=0), 'k', linewidth=thick_line_width,
                             label='mean')

        ax[-1].legend(loc='lower right', fontsize='small')

        for axis in ax:
            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)
            # axis.set_aspect(1)

        fig.savefig(f"{figures_path}/{group_name}/{fig_name}", bbox_inches='tight')
        if close_figure:
            plt.close(fig)


def optimize_theta_times(model_type='speed', plot=True, y_sigma=1.2, marker_size=0.4, close_figure=1):
    for group_name in group_names:
        paths = glob.glob(f"{pickles_path}/{sessions[0]}/{group_name}/PathLengths/"
                          f"single_cycles/{model_type}/predicted_positions*")
        theta_times = sorted([p.split('_')[-1] for p in paths])

        decoded_positions = np.empty(0)
        predicted_positions = [np.empty(0) for _ in range(len(theta_times))]

        for session in sessions:
            decoded_positions = np.append(decoded_positions,
                                          load(session, group_name, 'PathLengths/single_cycles/decoded_positions'))
            for theta_time_num, theta_time in enumerate(theta_times):
                rel_path = f'PathLengths/single_cycles/{model_type}/predicted_positions_{theta_time}'
                predicted_positions[theta_time_num] = np.append(predicted_positions[theta_time_num],
                                                                load(session, group_name, rel_path))

        slopes = []
        first_slopes = []
        for theta_time_num, theta_time in enumerate(theta_times):
            ok = (~np.isnan(decoded_positions) & ~np.isnan(predicted_positions[theta_time_num])
                  & ((predicted_positions[theta_time_num] > 0.01) | (predicted_positions[theta_time_num] < -0.01)))

            x = predicted_positions[theta_time_num][ok]
            y = decoded_positions[ok]
            iteration_slopes = []
            iteration_intercepts = []
            max_iterations = 10
            max_increment = 0.01
            previous_slope = np.nan
            for i in range(max_iterations):
                slope, intercept, r, p, e = linregress(x, y)
                iteration_slopes.append(slope)
                iteration_intercepts.append(intercept)
                errors = (decoded_positions[ok] - predicted_positions[theta_time_num][ok] * slope + intercept)**2
                mean_error_threshold = np.mean(errors) * 1.5
                x = predicted_positions[theta_time_num][ok][errors < mean_error_threshold]
                y = decoded_positions[ok][errors < mean_error_threshold]
                if abs(slope - previous_slope) < max_increment:
                    break
                previous_slope = slope
            slopes.append(slope)
            first_slopes.append(iteration_slopes[0])

            if plot:
                folder_path = f"{figures_path}/{group_name}/theta_time_optimisation"
                if not os.path.exists(folder_path):
                    os.mkdir(folder_path)
                fig, ax = plt.subplots()
                ax.plot(predicted_positions[theta_time_num][ok], decoded_positions[ok]
                        + np.random.normal(scale=y_sigma, size=np.sum(ok)), '.', markersize=marker_size, alpha=0.3)
                ax.plot(x, y + np.random.normal(scale=y_sigma, size=len(y)), '.', markersize=marker_size, alpha=0.3)
                x = np.array((np.nanmin(predicted_positions), np.nanmax(predicted_positions)))
                ax.plot(x, x * iteration_slopes[0] + iteration_intercepts[0],
                        label=f"first slope = {iteration_slopes[0]:.2f}")
                ax.plot(x, x * slope + intercept, label=f"slope = {slope:.2f}")
                ax.legend(loc='lower right')
                ax.set_xlim([np.nanpercentile(predicted_positions[theta_time_num], 1),
                             np.nanpercentile(predicted_positions[theta_time_num], 99)])
                ax.set_xlabel("Predicted position")
                ax.set_ylabel("Decoded position")
                fig.savefig(f"{folder_path}/{theta_time}.png", bbox_inches='tight')
                if close_figure:
                    plt.close(fig)

        fig, ax = plt.subplots()
        ax.plot([float(t) for t in theta_times], first_slopes, label='first slopes')
        ax.plot([float(t) for t in theta_times], slopes, label='iterated slopes')
        ax.axhline(1, linestyle='dashed', color='gray')
        ax.legend()
        ax.set_xlabel('Theta time (s)')
        ax.set_ylabel("Slope of decoded position vs predicted position")
        fig.savefig(f"{folder_path}/decoded_vs_predicted_slopes.png", bbox_inches='tight')
        if close_figure:
            plt.close(fig)

        best_index = np.argmin(np.abs(np.array(slopes) - 1))
        print(f'group_name = {group_name}; best theta_time = {theta_times[best_index]}')


def speed_bin_ok_ratios(path, hist_bin_size=10, fig_size=(2.6, 3.3), close_figure=True):
    for group_name in group_names:
        num_speed_bin_distances = len(load(sessions[0], group_name, path))
        rats_ok_bins = [[[] for _ in range(num_speed_bin_distances)] for _ in range(len(rats))]
        for session in sessions:
            rat_index = rats.index(session.split('.')[0])
            rat_ok_bins = load(session, group_name, path)
            for distance_num, distance_ok_bins in enumerate(rat_ok_bins):
                rats_ok_bins[rat_index][distance_num] += distance_ok_bins

        rat_means = []
        x = hist_bin_size * np.arange(num_speed_bin_distances) + hist_bin_size/2
        fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        for rat_index, (rat, rat_ok_bins) in enumerate(zip(rats, rats_ok_bins)):
            rat_mean = [np.mean(ok_bins)*100 for ok_bins in rat_ok_bins]
            rat_means.append(rat_mean)
            ax.plot(x, rat_mean, '.', label=rat, color=colors[rat_index])
        ax.set_xticks(x)
        ax.bar(x, np.mean(rat_means, axis=0), width=hist_bin_size, color='lightgray')
        ax.set_ylabel('Fields sufficiently sampled\nfor field analyses (%)')
        ax.set_xlabel("Deviation between\nspeed bin and\nfield's characteristic\nspeed (cm/s)")
        # ax.legend(loc='upper right', fontsize='x-small')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

        folder_path = f"{figures_path}/{group_name}"
        fig.savefig(f"{folder_path}/ok_bin_ratios", bbox_inches='tight')
        if close_figure:
            plt.close(fig)


def phase_precession_per_acceleration(positions_path, phases_path, speeds_path, accelerations_path,
                                      rel_speed_changes_path,
                                      circular_mean_path, mean_ranges_path, acceleration_groups, num_position_bins,
                                      num_phase_bins, logarithm=False, histogram_fig_size=(8, 3), fields_per_row=8,
                                      rows_per_figure=8, batch_fig_size=(11, 9), close_figures=True):

    position_bin_size = 1 / (num_position_bins - 1)
    phase_bin_size = 360 / (num_phase_bins - 1)

    for group_name in group_names:
        # create containers
        histograms = np.zeros((len(acceleration_groups), num_phase_bins, num_position_bins))
        all_rats = []
        all_positions = []
        all_phases = []
        all_speeds = []
        all_accelerations = []
        all_rel_speed_changes = []
        all_circular_means = []
        all_mean_ranges = []

        for session in sessions:
            # load the data
            positions = load(session, group_name, positions_path)
            phases = load(session, group_name, phases_path)
            speeds = load(session, group_name, speeds_path)
            accelerations = load(session, group_name, accelerations_path)[0]
            rel_speed_changes = load(session, group_name, rel_speed_changes_path)[0]
            circular_means = load(session, group_name, circular_mean_path)
            mean_ranges = load(session, group_name, mean_ranges_path)

            for field_positions, field_phases, field_rel_speed_changes in zip(positions, phases, rel_speed_changes):

                # half = int(len(field_speeds)/2)
                # speed_changes.append((np.nanmean(field_speeds[half+1:]) - np.nanmean(field_speeds[:half]))
                #                      / np.nanmean(field_speeds))

                for group_num, acceleration_group in enumerate(acceleration_groups):
                    if acceleration_group[0] <= field_rel_speed_changes < acceleration_group[1]:
                        for position, phase in zip(field_positions, field_phases):
                            position_bin = int(round(position/position_bin_size))
                            phase_bin = int(round(phase/phase_bin_size))
                            histograms[group_num, phase_bin, position_bin] += 1

            all_positions += positions
            all_phases += phases
            all_speeds += speeds
            all_rel_speed_changes += rel_speed_changes
            all_accelerations += accelerations
            rat = session.split('.')[0]
            all_rats += [rat for _ in range(len(accelerations))]
            all_circular_means += circular_means
            all_mean_ranges += mean_ranges

        fig, axes = plt.subplots(1, len(acceleration_groups), sharey='row', figsize=histogram_fig_size)

        for ax, histogram, acceleration_group in zip(axes, histograms, acceleration_groups):
            if logarithm:
                histogram += 1
                norm = colors_lib.LogNorm(vmin=1)
            else:
                norm = colors_lib.Normalize(vmin=0)
            ax.matshow(histogram, origin='lower', norm=norm, aspect='auto',
                       extent=(-position_bin_size/2, 1 + position_bin_size/2, -phase_bin_size/2, 360+phase_bin_size/2))
            ax.xaxis.set_ticks_position("bottom")
            ax.set_title(f"{acceleration_group}")

        folder_path = f"{figures_path}/{group_name}/acceleration"
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
        fig.savefig(f"{folder_path}/histograms")
        if close_figures:
            plt.close(fig)

        figs = []
        axes = []
        fields_per_figure = rows_per_figure*fields_per_row
        for _ in range(int(np.ceil(len(all_accelerations)/fields_per_figure))):
            fig, ax = plt.subplots(rows_per_figure, fields_per_row, sharex='all', sharey='all', figsize=batch_fig_size)
            figs.append(fig)
            axes.append(ax)

        # sorted_indices = np.argsort(all_accelerations)
        sorted_indices = np.argsort(all_rel_speed_changes)

        max_speed = max([max(field_speeds) for field_speeds in all_speeds])

        for field_num, sorted_index in enumerate(sorted_indices):
            field_num_within = field_num % fields_per_figure
            ax = axes[int(field_num/fields_per_figure)][int(field_num_within / fields_per_row),
                                                        field_num_within % fields_per_row]
            for mean_range in all_mean_ranges[sorted_index]:
                ax.axvline(mean_range[0], color='C7', linewidth=0.8)
                ax.axvline(mean_range[1], color='C7', linewidth=0.8)
            alpha = 0.8*np.exp(-0.02*len(all_positions[sorted_index])) + 0.2
            ax.plot(all_positions[sorted_index], all_phases[sorted_index], '.', alpha=alpha, markersize=3)
            ax.annotate(f"{all_accelerations[sorted_index]:.2f}", (0.6, 0.6), fontsize="x-small",
                        xycoords='axes fraction')
            ax.annotate(f"{all_rel_speed_changes[sorted_index]:.2f}", (0.6, 0.8), fontsize="x-small",
                        xycoords='axes fraction')

            ax_twin = ax.twinx()
            field_speeds = all_speeds[sorted_index]
            ax_twin.plot(np.linspace(0, 1, len(field_speeds)), field_speeds, color='C7', linewidth=1)
            ax_twin.set_ylim((0, max_speed))
            for key in ['left', 'top', 'right', 'bottom']:
                ax_twin.spines[key].set_edgecolor(colors[rats.index(all_rats[sorted_index])])

            x = [(mean_range[0] + mean_range[1]) / 2 for mean_range in all_mean_ranges[sorted_index]]
            ax.plot(x, all_circular_means[sorted_index], '*-', color='k')

        for fig_num, fig in enumerate(figs):
            fig.tight_layout()
            fig.savefig(f"{figures_path}/{group_name}/acceleration/batch {fig_num}")
            if close_figures:
                plt.close(fig)


def significant_cycles():
    values_container = [[] for _ in rats]

    for session in sessions:
        rat_index = rats.index(session.split('.')[0])
        rat_values = load(session, None, "LFP/significant_ratios")
        values_container[rat_index] += rat_values

    for rat_index, rat in enumerate(rats):
        print(f"{rat}: mean = {np.mean(values_container[rat_index])}, std = {np.std(values_container[rat_index])}")
    print(f"ALL: mean = {np.mean(np.concatenate(values_container))}, std = {np.std(np.concatenate(values_container))}")

    print(1)





