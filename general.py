import os
import pathlib
import pickle
import inspect
import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from scipy.stats import linregress
import matplotlib.pyplot as plt


class Base:
    """Base class from which to derive the other analysis classes. Includes functions for:
        * Loading instantiated objects from pickles when appropriate.
        * Saving figures more easily and homogeneously.

    Args:
        super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is defined
            as belonging to the super-group, it will be shared across sub-groups.
        group_name (string): Name of the low-level sub-group used for pickles and figures.
        child_name (string): Name of the instance used for pickles and figures.
        save_figures (bool): Whether to save and close the figures.
        figure_format (string): Format of the figures (e.g., png).
        pickle_results (bool): Whether to pickle the results.

    Attributes:
        figures_path (string): Path to the figures folder.
        pickles_path (string): Path to the folder where results will be pickled.
    """
    dependencies = ()
    belongs_to_super_group = False

    def __init__(self, super_group_name, group_name, child_name, save_figures=False, figure_format="png",
                 figures_path="figures", pickle_results=True, pickles_path="pickles"):
        self.super_group_name = super_group_name
        self.group_name = group_name
        self.child_name = child_name

        self.relative_path = f"{super_group_name}/"
        if not self.belongs_to_super_group:
            self.relative_path += f"{group_name}/"
        self.relative_path += f"{child_name}/"

        self.save_figures = save_figures
        self.figure_format = figure_format
        self.figures_path = self.build_complete_path(figures_path)

        self.pickle_results = pickle_results
        self.pickles_path = self.build_complete_path(pickles_path)

    def build_complete_path(self, path):
        return f"{path}/{self.relative_path}"

    def maybe_save_fig(self, fig, fig_name, subfolder="", dpi=200):
        """Save figure if 'save_figures' is True.

        Args:
            fig: Matplotlib figure.
            fig_name (string): Name for the file.
            subfolder (string): Name for the sub-folder.
            dpi (int): Dots per inch.
        """
        if self.save_figures:
            folder_path = self.figures_path
            if subfolder:
                folder_path += subfolder + "/"
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)
            file_path = f"{folder_path}{fig_name}.{self.figure_format}"
            fig.savefig(file_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    def maybe_pickle_results(self, results, name, subfolder=""):
        """Pickle results if 'pickle_results' is True.

        Args:
            results: Python object structure to be pickled.
            name (string): Name for the file.
            subfolder (string): Name for the sub-folder.
        """
        if self.pickle_results:
            folder_path = self.pickles_path
            if subfolder:
                folder_path += subfolder + "/"
            if not os.path.isdir(folder_path):
                pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)

            file_path = f"{folder_path}{name}"
            with open(file_path, 'wb') as f:
                pickle.dump(results, f)

    @classmethod
    def default_initialization(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                               figure_format="png", figures_path="", pickle_results=False, pickles_path="", **kwargs):
        """Method that will be called to initialize a new instance of the class when necessary.

        Args:
            super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is
                defined as belonging to the super-group, it will be shared across sub-groups.
            group_name (string): Name of the low-level sub-group used for pickles and figures.
            child_name (string): Name of the instance used for pickles and figures.
            parameters_dict (dict): Dictionary of parameters.
            save_figures (bool): Whether to save and close the figures.
            figure_format (string): Matplotlib figure format (e.g., "png", "pdf", etc.)
            figures_path (string): Path to the folder where figures will be saved.
            pickle_results (bool): Whether to allow the pickling of results.
            pickles_path (string): Path to the folder where pickles will be saved.
            kwargs (dict): Additional keyed arguments.
        """
        pass

    @classmethod
    def default_pickle(cls, super_group_name, group_name, child_name, parameters_dict, save_figures=False,
                       figure_format="png", figures_path="figures", pickle_results=True, pickles_path="pickles",
                       **kwargs):
        """Runs the default_initialization function for creating an new instance of the object if something has been
        changed, otherwise loads a previous instance from a pickle.

        Args:
            super_group_name (string): Name of the high-level group used for pickles and figures. If an instance is
                defined as belonging to the super-group, it will be shared across sub-groups.
            group_name (string): Name of the low-level sub-group used for pickles and figures.
            child_name (string): Name of the instance used for pickles and figures.
            parameters_dict (dict): Dictionary of parameters used by the initialization function.
            save_figures (bool): Whether to save and close the figures.
            figure_format (string): Matplotlib figure format (e.g., "png", "pdf", etc.)
            figures_path (string): Path to the folder where figures will be saved.
            pickle_results (bool): Whether to allow the pickling of results.
            pickles_path (string): Path to the folder where pickles will be saved.

            kwargs (dict): Additional keyed arguments for the default initialization function.

        Returns:
            (tuple): A tuple containing:
                * instance (object): Initialized instance.
                * instance_is_new (bool): Whether the instance has just been created.
        """
        def dependency_instance_pickle_path(dependency):
            if dependency.belongs_to_super_group:
                return f"{pickles_path}/{super_group_name}/{dependency.__name__}/instance"
            else:
                return f"{pickles_path}/{super_group_name}/{group_name}/{dependency.__name__}/instance"

        def get_latest_times():
            latest_times = {}
            for dependency in cls.dependencies:
                latest_times[dependency.__name__] = os.path.getmtime(dependency_instance_pickle_path(dependency))
            return latest_times

        child_pickles_path = pickles_path + f"/{super_group_name}/"
        if not cls.belongs_to_super_group:
            child_pickles_path += f"{group_name}/"
        child_pickles_path += child_name

        classes_changed = False
        for my_class in inspect.getmro(cls)[:-1]:
            classes_changed += cls.changed(f"{child_pickles_path}/{my_class.__name__}", my_class, inspect.getsource)

        parameters_changed = cls.changed(f"{child_pickles_path}/params", parameters_dict)

        # compare the modification dates of dependency instances with those used in the previous initialization
        need_new = True
        instance_pickle_path = f"{child_pickles_path}/instance"
        times_pickle_path = f"{child_pickles_path}/times"
        if (os.path.exists(instance_pickle_path) and os.path.exists(times_pickle_path)
                and not classes_changed and not parameters_changed):
            need_new = False
            with open(times_pickle_path, 'rb') as times_f:
                if pickle.load(times_f) != get_latest_times():
                    need_new = True

        if need_new:
            print(f"Initializing {child_pickles_path}...")

            if not os.path.isdir(child_pickles_path):
                os.makedirs(child_pickles_path)

            # load dependencies into kwargs
            for dependency in cls.dependencies:
                with open(dependency_instance_pickle_path(dependency), 'rb') as instance_f:
                    kwargs[dependency.__name__] = pickle.load(instance_f)

            # initialize and dump instance and times
            instance = cls.default_initialization(super_group_name, group_name, child_name, parameters_dict,
                                                  save_figures, figure_format, figures_path, pickle_results,
                                                  pickles_path, **kwargs)

            with open(instance_pickle_path, 'wb') as instance_f:
                pickle.dump(instance, instance_f)
            with open(times_pickle_path, 'wb') as times_f:
                pickle.dump(get_latest_times(), times_f)

            print("... done!")

        else:
            print(f"Loading {child_pickles_path}...")
            with open(instance_pickle_path, 'rb') as instance_f:
                instance = pickle.load(instance_f)

            instance.pickle_results = pickle_results
            instance.pickles_path = instance.build_complete_path(pickles_path)
            instance.save_figures = save_figures
            instance.figure_format = figure_format
            instance.figures_path = instance.build_complete_path(figures_path)

        return instance

    @staticmethod
    def changed(path, something, extract_relevant=lambda x: x):
        """Checks whether something matches with a previous pickled version. If there
        is no match, a new pickle is created.

        Args:
            path (string): Path where pickles will be stored.
            something (object): Function, class, built-in type, etc. to compare.
            extract_relevant (function): A function that extracts the relevant part to be compared.
        Returns:
            (bool): Whether the source code matched that of a previous pickled version.
        """
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                if extract_relevant(something) == pickle.load(f):
                    return False

        folder = path[:len(path) - path[::-1].index('/')]
        if not os.path.exists(folder):
            pathlib.Path(folder).mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(extract_relevant(something), f)
        return True


def interpolate_missing(values):
    """Interpolate missing values in a vector of values with nan entries.

    Args:
        values (np.array): 1D numpy array of values.
    """
    not_nan = ~np.isnan(values)
    interpolating_function = interp1d(np.arange(values.size)[not_nan], values[not_nan])
    interpolation_range = slice(int(np.argmax(not_nan)), values.size - int(np.argmax(not_nan[::-1])))
    interpolated = values[interpolation_range].copy()
    is_nan = np.isnan(interpolated)
    interpolated[is_nan] = interpolating_function(np.arange(values.size)[interpolation_range][is_nan])
    return interpolated, interpolation_range


def nan_smooth(values, sigma, mode='nearest'):
    """Apply a Gaussian filter to a vector of values with nan entries.

    Args:
        values (np.array): 1D numpy array of values.
        sigma (float): Sigma for the Gaussian filter.
        mode (string): 'mode'parameter of scipy.ndimage.filters.gaussian_filter1d
    """
    if sigma == 0:
        return values
    else:
        interpolated, interpolation_range = interpolate_missing(values)
        smoothed = np.full(values.size, np.nan)
        smoothed[interpolation_range] = gaussian_filter1d(interpolated, sigma, mode=mode)
        return smoothed


def radon_fit(x, y, num_slopes, slope_bounds, num_intercepts, intercept_bounds, d):
    best_sum = 0
    best_slope = 0
    best_intercept = 0

    for slope in np.linspace(slope_bounds[0], slope_bounds[1], num_slopes):
        d_y = d * np.sqrt(1 + slope ** 2)
        for intercept in np.linspace(intercept_bounds[0], intercept_bounds[1], num_intercepts):
            ok = (y >= slope * x + intercept - d_y) & (y <= slope * x + intercept + d_y)
            current_sum = np.sum(ok)
            if current_sum > best_sum:
                best_sum = current_sum
                best_slope = slope
                best_intercept = intercept

    return best_slope, best_intercept



