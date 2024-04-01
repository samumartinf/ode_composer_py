from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    WhiteKernel,
)
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from collections import defaultdict
from sklearn.model_selection import LeaveOneOut
from .util import Timer
from .cache_manager import CacheManager


class BatchSignalPreprocessor(object):
    """
    A class for batch signal preprocessing.

    Parameters:
    - t (array-like): The time values.
    - data (dict): A dictionary containing the signal data.
    - method (str): The method to use for preprocessing.

    Attributes:
    - prepocessors (dict): A dictionary containing the signal preprocessors.

    Methods:
    - interpolate(t_new): Interpolates the signal data at new time values.
    - calculate_time_derivative(t_new): Calculates the time derivative of the signal data at new time values.
    """

    def __init__(self, t, data, method):
        self.prepocessors = defaultdict()
        if method == "SplineSignalPreprocessor":
            for key, datum in data.items():
                self.prepocessors[key] = SplineSignalPreprocessor(t, datum)

    def interpolate(self, t_new):
        """
        Interpolates the signal data at new time values.

        Parameters:
        - t_new (array-like): The new time values.

        Returns:
        - ret_data (dict): A dictionary containing the interpolated signal data.
        """
        ret_data = defaultdict()
        for key, preprocessor in self.prepocessors.items():
            ret_data[key] = preprocessor.interpolate(t_new)
        return ret_data

    def calculate_time_derivative(self, t_new):
        """
        Calculates the time derivative of the signal data at new time values.

        Parameters:
        - t_new (array-like): The new time values.

        Returns:
        - ret_data (dict): A dictionary containing the calculated time derivatives of the signal data.
        """
        ret_data = defaultdict()
        for key, preprocessor in self.prepocessors.items():
            ret_data[key] = preprocessor.calculate_time_derivative(t_new)
        return ret_data


class SignalPreprocessor(object):
    """
    A class for preprocessing signals.

    Attributes:
        t (array-like): The time values of the signal.
        y (array-like): The signal values.
    """

    def __init__(self, t, y):
        self._dydt = None
        self.y = y
        self.t = t

    @property
    def dydt(self):
        """
        Get the derivative of the signal.

        Returns:
            array-like: The derivative of the signal.
        """
        # TODO add checks
        return self._dydt

    @dydt.setter
    def dydt(self, new_value):
        raise ValueError(
            "dydt cannot be changed from the outside of the object!"
        )


class GPSignalPreprocessor(SignalPreprocessor):
    """
    A class for preprocessing signals using Gaussian Process regression.

    Parameters:
    - t (array-like): The time values of the signal.
    - y (array-like): The observed values of the signal.
    - selected_kernel (str, optional): The selected kernel for Gaussian Process regression. Defaults to "RatQuad".
    - interpolation_factor (int, optional): The factor by which to extend the time range for interpolation. Defaults to None.

    Attributes:
    - kernels (dict): A dictionary of different kernels that will be explored.
    - selected_kernel (str): The selected kernel for Gaussian Process regression.
    - interpolation_factor (int): The factor by which to extend the time range for interpolation.
    - A_mean (array-like): The mean values of the interpolated signal.
    - A_std (array-like): The standard deviation values of the interpolated signal.
    - noisy_kernels (dict): A dictionary of noisy kernels generated from the selected kernel.

    Methods:
    - interpolate(return_extended_time=False, noisy_obs=True): Interpolates the signal using Gaussian Process regression.
    - calculate_time_derivative(): Calculates the time derivative of the interpolated signal.
    - diff_matrix(size): Generates a differentiation matrix used as a linear operator.

    """

    def __init__(
        self, t, y, selected_kernel="RatQuad", interpolation_factor=None
    ):
        super().__init__(t, y)
        self.kernels = None
        self.selected_kernel = selected_kernel
        self.interpolation_factor = interpolation_factor

        # TODO: fix this to comply with python standards
        self.A_mean = None
        self.A_std = None

        # Create different kernels that will be explored
        self.kernels = dict()

        self.kernels["RBF"] = 1.0 * RBF(length_scale=0.5)
        self.kernels["RatQuad"] = 1.0 * RationalQuadratic(
            length_scale=1.0, alpha=0.2
        )
        self.kernels["ExpSineSquared"] = 1.0 * ExpSineSquared(
            length_scale=1.0, periodicity=3
        )
        self.kernels["Matern"] = 1.0 * Matern(length_scale=1.0, nu=1.5)

        self.kernels["Matern*ExpSineSquared"] = (
            1.0
            * Matern(length_scale=1.0, nu=1.5)
            * ExpSineSquared(length_scale=1, periodicity=3)
        )

        self.kernels["RBF*ExpSineSquared"] = (
            1.0
            * RBF(length_scale=1.0)
            * ExpSineSquared(length_scale=1, periodicity=3)
        )

        self.kernels["RatQuad*ExpSineSquared"] = (
            1.0
            * RationalQuadratic(length_scale=1.0, alpha=0.2)
            * ExpSineSquared(length_scale=1, periodicity=3)
        )

        self.kernels["Matern*RBF"] = (
            1.0 * Matern(length_scale=1.0, nu=1.5) * RBF(length_scale=1)
        )

        self.kernels["Matern+ExpSineSquared"] = 1.0 * Matern(
            length_scale=1.0, nu=1.5
        ) + ExpSineSquared(length_scale=1, periodicity=3)

        self.kernels["RBF+ExpSineSquared"] = 1.0 * RBF(
            length_scale=1.0
        ) + ExpSineSquared(length_scale=1, periodicity=3)

        self.kernels["RatQuad+ExpSineSquared"] = 1.0 * RationalQuadratic(
            length_scale=1.0
        ) + ExpSineSquared(length_scale=1, periodicity=3)

        if selected_kernel not in self.kernels.keys():
            raise KeyError(
                f"Unknown kernel: {selected_kernel}, available kernels: {self.kernels.keys()}"
            )

        # Generate the noisy kernels
        self.noisy_kernels = dict()
        for key, kernel in self.kernels.items():
            self.noisy_kernels[key] = kernel + WhiteKernel(
                noise_level=1, noise_level_bounds=(1e-7, 1e7)
            )
    
    def get_avaialble_kernels(self):
        """
        Get the available kernels for Gaussian Process regression.

        Returns:
        - array-like: The available kernels.
        """
        return self.kernels.keys()

    def interpolate(self, return_extended_time=False, noisy_obs=True):
        """
        Interpolates the signal using Gaussian Process regression.

        Parameters:
        - return_extended_time (bool, optional): Whether to return the extended time range. Defaults to False.
        - noisy_obs (bool, optional): Whether to use noisy observations for interpolation. Defaults to True.

        Returns:
        - A_mean (array-like): The mean values of the interpolated signal.
        - X_extended (array-like, optional): The extended time range if return_extended_time is True, otherwise the original time range.

        """
        # Adjust the number of samples to be drawn from the fitted GP

        if noisy_obs:
            actual_kernel = self.noisy_kernels[self.selected_kernel]
        else:
            actual_kernel = self.kernels[self.selected_kernel]

        gp = GaussianProcessRegressor(kernel=actual_kernel)

        X = self.t[:, np.newaxis]
        gp.fit(X, self.y)

        if self.interpolation_factor is None or self.interpolation_factor == 1:
            self.A_mean, self.A_std = gp.predict(X, return_std=True)
            _, self.K_A = gp.predict(X, return_cov=True)
        elif self.interpolation_factor > 0:
            X_extended = np.linspace(
                self.t[0], self.t[-1], self.interpolation_factor * len(self.t)
            )
            X_extended = X_extended[:, np.newaxis]
            self.A_mean, self.A_std = gp.predict(X_extended, return_std=True)
            _, self.K_A = gp.predict(X_extended, return_cov=True)
        else:
            raise KeyError(
                "Please ensure the interpolation factor is a positive number"
            )

        if return_extended_time and self.interpolation_factor is not None:
            X_extended = np.linspace(
                self.t[0], self.t[-1], self.interpolation_factor * len(self.t)
            )
            return self.A_mean, X_extended
        else:
            return self.A_mean, self.t

    def calculate_time_derivative(self):
        """
        Calculates the time derivative of the interpolated signal.

        """
        dA_mean = np.gradient(self.A_mean)
        if self.interpolation_factor is None:
            dTime = np.gradient(self.t)
        else:
            t_extended = np.linspace(
                self.t[0], self.t[-1], self.interpolation_factor * len(self.t)
            )
            dTime = np.gradient(t_extended)

        dA_mean = dA_mean / dTime

        self._dydt = dA_mean

    def diff_matrix(self, size):
        """
        Generates a differentiation matrix used as a linear operator.

        Parameters:
        - size (int): The size of the differentiation matrix.

        Returns:
        - A (array-like): The differentiation matrix.

        """
        A = np.zeros((size, size))
        b = np.ones(size - 1)
        np.fill_diagonal(A[0:], -b)
        np.fill_diagonal(A[:, 1:], b)
        return A


class SplineSignalPreprocessor(SignalPreprocessor):
    """
    A class that represents a signal preprocessor using cubic spline interpolation.

    Attributes:
        t (array-like): The time values of the input signal.
        y (array-like): The corresponding signal values.
        cs (CubicSpline): The cubic spline object used for interpolation.

    Methods:
        interpolate(t_new): Interpolates the signal at new time values.
        calculate_time_derivative(t_new): Calculates the time derivative of the signal at new time values.
    """

    def __init__(self, t, y, **kwargs):
        super().__init__(t, y)
        self.cs = None

    def interpolate(self, t_new):
        """
        Interpolates the signal at new time values using cubic spline interpolation.

        Args:
            t_new (array-like): The new time values to interpolate the signal at.

        Returns:
            array-like: The interpolated signal values at the new time values.
        """
        self.cs = CubicSpline(self.t, self.y)
        return self.cs(t_new)

    def calculate_time_derivative(self, t_new):
        """
        Calculates the time derivative of the signal at new time values.

        Args:
            t_new (array-like): The new time values to calculate the time derivative at.

        Returns:
            array-like: The time derivative of the signal at the new time values.
        """
        if self.cs is None:
            self.interpolate(t_new=t_new)

        pp = self.cs.derivative()
        return pp(t_new)


class RHSEvalSignalPreprocessor(SignalPreprocessor):
    """
    A signal preprocessor that evaluates the right-hand side (RHS) function to calculate the time derivative.

    Args:
        t (array-like): The time values.
        y (array-like): The state values.
        rhs_function (callable): The function that calculates the RHS of the ODE system.
        states (dict): Additional states required by the RHS function.

    Attributes:
        rhs_function (callable): The function that calculates the RHS of the ODE system.
        states (dict): Additional states required by the RHS function.

    Methods:
        interpolate(): Interpolates the signal.
        calculate_time_derivative(): Calculates the time derivative using the RHS function.

    """

    def __init__(self, t, y, rhs_function, states):
        super().__init__(t, y)
        self.rhs_function = rhs_function
        self.states = states

    def interpolate(self):
        pass

    def calculate_time_derivative(self):
        rr = list()
        for yy in self.y.T:
            rr.append(self.rhs_function(0, yy, self.states))

        self._dydt = np.array(rr).T


class ZeroOrderHoldPreprocessor(SignalPreprocessor):
    """
    A signal preprocessor that implements the zero-order hold interpolation method.

    This preprocessor calculates the time derivative and performs interpolation using the zero-order hold method.
    """

    def __init__(self, t, y):
        super(ZeroOrderHoldPreprocessor, self).__init__(t=t, y=y)

    def calculate_time_derivative(self):
        raise NotImplementedError(
            "Time derivative calculation is not implemented for Zero order hold!"
        )

    def interpolate(self, t_new):
        """
        Interpolates the signal at the given time points using the zero-order hold method.

        Args:
            t_new (float or array-like): The time points at which to interpolate the signal.

        Returns:
            The interpolated signal values at the given time points.
        """
        # TODO ZAT support non pandas data format too!
        ret = []
        if isinstance(t_new, float):
            return self.y[abs(self.t - t_new).idxmin()]

        for t_i in t_new:
            ret.append(self.y[abs(self.t - t_i).idxmin()])
        return ret


class SmoothingSplinePreprocessor(SignalPreprocessor):
    """
    A class that represents a signal preprocessor using smoothing splines.

    Parameters:
    - t (array-like): The time values of the signal.
    - y (array-like): The corresponding signal values.
    - tune_smoothness (bool, optional): Whether to tune the smoothness parameter. Defaults to True.
    - weights (array-like, optional): The weights for the data points. Defaults to None.
    - spline_id (str, optional): The identifier for caching the spline data. Defaults to None.
    - cache_folder (str, optional): The folder path for caching the spline data. Defaults to None.
    """

    def __init__(
        self,
        t,
        y,
        tune_smoothness=True,
        weights=None,
        spline_id=None,
        cache_folder=None,
    ):
        super().__init__(t, y)
        self.cs = None
        self.s = None
        self.weights = weights
        self.spline_id = spline_id
        self.cache_folder = cache_folder

        if tune_smoothness:
            self.tune_smoothness()

    def _cache_checked(f):
        def cache_checker(*args, **kwargs):
            self = args[0]
            cache_manager = CacheManager(
                cache_id=self.spline_id, cache_folder=self.cache_folder
            )
            if cache_manager.cache_hit():
                cached_data = cache_manager.read()
                self.s = cached_data["smoothness"]
            else:
                f(*args, **kwargs)
                data_to_cache = {"smoothness": self.s}
                cache_manager.write(data_to_cache)

        return cache_checker

    @_cache_checked
    def tune_smoothness(self):
        """
        Tune the smoothness parameter using cross-validation.

        This method uses leave-one-out cross-validation to find the optimal smoothness parameter
        for the smoothing spline.

        Returns:
        - None
        """

        sum_res = [[] for _ in range(len(self.t))]

        loo = LeaveOneOut()
        sweep = [0] + list(np.logspace(-4, 4, 100)) + [len(self.t)]
        with Timer(
            f"CV loop for SmoothingSplinePreprocessor on {self.spline_id}"
        ):
            for case_idx, (train_index, test_index) in enumerate(
                loo.split(self.t)
            ):
                if self.weights is not None:
                    w = self.weights.iloc[train_index]
                else:
                    w = None
                X_train, X_test = (
                    self.t.iloc[train_index],
                    self.t.iloc[test_index],
                )
                y_train, y_test = (
                    self.y.iloc[train_index],
                    self.y.iloc[test_index],
                )
                spl = UnivariateSpline(X_train, y_train, w=w)

                for s in sweep:
                    spl.set_smoothing_factor(s=s)
                    sum_res[case_idx].append(
                        np.square(float(y_test - spl(X_test)))
                    )

        total = np.sum(np.array(sum_res), axis=0)

        s_opt_idx = np.argmin(total)
        self.s = sweep[s_opt_idx]

    def interpolate(self, t_new):
        """
        Interpolate the signal at new time points.

        Parameters:
        - t_new (array-like): The new time points to interpolate the signal at.

        Returns:
        - array-like: The interpolated signal values at the new time points.
        """

        self.cs = UnivariateSpline(self.t, self.y, s=self.s, w=self.weights)

        return self.cs(t_new)

    def calculate_time_derivative(self, t_new):
        """
        Calculate the time derivative of the signal at new time points.

        Parameters:
        - t_new (array-like): The new time points to calculate the time derivative at.

        Returns:
        - array-like: The time derivative of the signal at the new time points.
        """

        if self.cs is None:
            self.interpolate(t_new=t_new)

        pp = self.cs.derivative()
        return pp(t_new)
