import copy
import os
import cvxpy as cp
import numpy as np
import scipy as sci
from typing import List, Dict, Union
from .linear_model import LinearModel
from .dictionary_builder import MultiVariableFunction
from .errors import SBLError
from .util import Timer
import warnings
import logging
from sympy import diff
from statsmodels.tsa.stattools import adfuller
from ode_composer.dictionary_builder import DictionaryBuilder

logger = logging.getLogger(f"ode_composer_{os.getpid()}")


class SBL(object):
    """
    Sparse Bayesian Learning (SBL) class for estimating model parameters.

    Args:
        dict_mtx (np.ndarray): The dictionary matrix.
        data_vec (np.ndarray): The data vector.
        lambda_param (float): The lambda parameter.
        dict_fcns (List[MultiVariableFunction], optional): List of dictionary functions. Defaults to None.
        state_name (str, optional): The name of the state. Defaults to None.
        config (Dict, optional): Configuration settings. Defaults to None.

    Attributes:
        linear_model: The linear model.
        z: The parameter z.
        w_estimates (List[float]): List of estimated weights.
        z_estimates (List[float]): List of estimated z values.
        gamma_estimates (List[float]): List of estimated gamma values.
        dict_fcns (List[MultiVariableFunction]): List of dictionary functions.
        lambda_param (float): The lambda parameter.
        state_name (str): The name of the state.
        config (Dict): Configuration settings.
        non_zero_idx: The indices of non-zero gamma values.
        solver_keywords (Dict): Dictionary of arguments for problem.solve.

    Properties:
        lambda_param (float): Getter and setter for the lambda parameter.
        dict_fcns: Getter and setter for the dictionary functions.
        config: Getter and setter for the configuration settings.

    Methods:
        data_fit: Computes the data fit.
        regularizer: Computes the regularizer.
        objective_fn: Computes the objective function.
        estimate_model_parameters: Estimates the model parameters.
        update_z: Computes the z value.
        compute_model_structure: Computes the model structure.
        compute_non_zero_idx: Computes the indices of non-zero gamma values.
        get_results: Gets the results.

    """

    def __init__(
        self,
        dict_mtx: np.ndarray,
        data_vec: np.ndarray,
        lambda_param: float,
        dict_fcns: List[MultiVariableFunction] = None,
        state_name: str = None,
        config: Dict = None,
    ):
        self.linear_model = LinearModel(dict_mtx, data_vec=data_vec)
        self.z = cp.Parameter(self.linear_model.parameter_num)
        self.z.value = np.ones(self.linear_model.parameter_num)
        self.w_estimates: List[float] = list()
        self.z_estimates: List[float] = list()
        self.gamma_estimates: List[float] = list()
        if dict_fcns:
            self.dict_fcns: List[MultiVariableFunction] = copy.deepcopy(
                dict_fcns
            )
        self.lambda_param = lambda_param
        self.state_name = state_name
        self.config = config
        self.non_zero_idx = None
        # build a dictionary of arguments for problem.solve
        self.solver_keywords = dict()
        self.solver_keywords["verbose"] = self.config["verbose"]
        self.solver_keywords["solver"] = self.config["solver"]["name"]
        self.solver_keywords.update(self.config["solver"]["settings"])

    @property
    def lambda_param(self) -> float:
        return self._lambda_param

    @lambda_param.setter
    def lambda_param(self, new_lambda_param: float):
        if not isinstance(new_lambda_param, float):
            new_lambda_param = float(new_lambda_param)
        if new_lambda_param < 0:
            raise ValueError(
                "lambda param must be non-negative, not %s!" % new_lambda_param
            )

        self._lambda_param = new_lambda_param

    @property
    def dict_fcns(self):
        return self._dict_fcns

    @dict_fcns.setter
    def dict_fcns(self, new_dict_fcns):
        if len(new_dict_fcns) == 0:
            raise ValueError("The dictionary functions cannot be empty!")
        self._dict_fcns = new_dict_fcns

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, new_config):
        # TODO ZAT if the config gets to complex, use https://github.com/clarketm/mergedeep
        self._config = {
            "solver": {"name": "ECOS", "show_time": False, "settings": {}},
            "verbose": False,
            "monitor_conv": False,
        }
        if new_config is not None:
            self._config.update(new_config)
            if "settings" not in self._config["solver"]:
                self._config["solver"]["settings"] = {}

    def data_fit(self, w):
        return (1.0 / 2.0) * cp.sum_squares(
            self.linear_model.dict_mtx @ w - self.linear_model.data_vec
        )

    def regularizer(self, w):
        z_sqrt = np.sqrt(self.z.value)
        w_abs = cp.atoms.elementwise.abs.abs(w)
        return z_sqrt @ w_abs

    def objective_fn(self, w):
        return self.data_fit(w) + 2 * self.lambda_param * self.regularizer(w)

    def estimate_model_parameters(self):
        w_variable = cp.Variable(self.linear_model.parameter_num)
        # add constraints to enforce non-negative ODEs solution
        constraints = []
        if (
            self.config is not None
            and "nonnegative" in self.config
            and self.state_name is not None
        ):
            for idx, d_f in enumerate(self.dict_fcns):
                if diff(d_f.symbolic_expression, self.state_name) == 0:
                    constraints.append(w_variable[idx] >= 0)

        problem = cp.Problem(
            objective=cp.Minimize(self.objective_fn(w=w_variable)),
            constraints=constraints,
        )
        try:
            problem.solve(**self.solver_keywords)

            if self.config["solver"]["show_time"]:
                logger.info(
                    f"solve time for {self.state_name}: {problem.solver_stats.solve_time}"
                )
            if problem.status == cp.OPTIMAL:
                # TODO update the underlying linear model with the new parameter
                self.w_estimates.append(w_variable.value)
            else:
                if problem.status == cp.OPTIMAL_INACCURATE:
                    warnings.warn(
                        f"Problem with optimization accuracy: {problem.status}"
                    )
                    self.w_estimates.append(w_variable.value)
                else:
                    print(
                        f"Problem with the solution from cvxpy: {problem.status}"
                    )
        except cp.error.SolverError as e:
            if "The solver" in str(e) and "is not installed." in str(e):
                raise cp.error.SolverError(e)
            else:
                warnings.warn(f"Solver encountered a problem: {e}")
            return False
        else:
            return True

    def update_z(self):
        """Computes z value by computing \gamma and z_{opt}

        updated \Sigma_y matrix
        \gamma_i = |w_i|/\sqrt{z_i},\qquad, i=1,\ldots,p
        \Gamma = diag[\gamma]
        \Sigma_y = \lambda I + A\Gamma A

        updated the z value
        z_{opt} = diag[A^\top \Sigma_y^{-1} A]

        Returns: z_{opt}
        """
        w_actual = self.w_estimates[-1]
        # update the underlying linear model
        self.linear_model.w = w_actual
        gamma = np.divide(abs(w_actual), np.sqrt(self.z.value.T)).T
        Gamma_diag = np.zeros((gamma.shape[0], gamma.shape[0]), float)
        np.fill_diagonal(Gamma_diag, gamma)
        Sigma_y = self.lambda_param * np.eye(
            self.linear_model.data_num
        ) + self.linear_model.dict_mtx @ Gamma_diag @ np.transpose(
            self.linear_model.dict_mtx
        )

        self.z.value = np.diag(
            np.transpose(self.linear_model.dict_mtx)
            @ np.linalg.solve(Sigma_y, self.linear_model.dict_mtx)
        )
        self.z_estimates.append(self.z.value)
        self.gamma_estimates.append(gamma)
        # print(f'state: {self.state_name} Sigma_y cond: {np.linalg.cond(np.linalg.pinv(Sigma_y, hermitian=True))} gamma: {np.divide(np.abs(w_actual),np.sqrt(self.z.value))}')

    def compute_model_structure(self, max_iter=10):
        """
        Computes the model structure using the Sparse Bayesian Learning (SBL) algorithm.

        Args:
            max_iter (int): The maximum number of iterations for the SBL algorithm. Defaults to 10.

        Raises:
            SBLError: If no SBL solution can be computed for the given state name.

        Returns:
            None
        """
        # TODO transform this into a generator
        # initialize convergence monitor
        conv_monitor = ConvergenceMonitor(SBL_problem=self)
        for idx in range(max_iter):
            if self.estimate_model_parameters():
                # model parameters were successfully estimated
                if self.lambda_param > 0:
                    self.update_z()
                    self.compute_non_zero_idx()
                if self.config["monitor_conv"]:
                    conv_monitor.calculate_convergence()
                    if conv_monitor.is_converged():
                        logger.info(
                            f"convergence threshold has been reached, SBL on {self.state_name} has stopped"
                        )
                        break
            else:
                # the solver encountered an error, let's see if partial results are available
                if idx > 0:
                    warnings.warn(
                        f"The solver encountered errors, only partial results are available after {idx+1} iterations."
                    )
                else:
                    raise SBLError(
                        f"No SBL solution can be computed for {self.state_name}!"
                    )

    def compute_non_zero_idx(self):
            """
            Compute the indices of non-zero elements based on the estimates of w and z. Sets the output to non_zero_idx.

            Returns:
                None
            """
            gamma = np.divide(
                np.abs(self.w_estimates[-1]), np.sqrt(self.z_estimates[-1])
            )
            tmp = np.nonzero(gamma > np.finfo(float).eps)
            self.non_zero_idx = tmp[0]

    def get_results(
            self, zero_th: float = None
        ) -> List[MultiVariableFunction]:
            """
            Returns the results of the SBL object.

            Args:
                zero_th (float, optional): Threshold value for considering weights as zero. Defaults to None.

            Returns:
                List[MultiVariableFunction]: List of dictionary functions with estimated weights.
            """
            zero_idx = list()
            if len(self.w_estimates) == 0:
                raise ValueError(
                    "SBL object contains no results! Did you run compute_model_structure()?"
                )

            w_est = self.w_estimates[-1]
            # update the dictionary functions with the estimated weights
            for d_f, w in zip(self.dict_fcns, w_est):
                d_f.constant = w

            if zero_th is not None:
                zero_idx = [
                    idx for idx, w in enumerate(w_est) if abs(w) <= zero_th
                ]

            if len(zero_idx) > 0:
                d_fcns = copy.deepcopy(self.dict_fcns)
                for idx in sorted(zero_idx, reverse=True):
                    del d_fcns[idx]
                return d_fcns
            else:
                return self.dict_fcns


class BatchSBL(object):
    def __init__(
        self,
        dict_mtx: Union[np.ndarray, List[np.ndarray]],
        data_vec: Union[np.ndarray, List[np.ndarray]],
        lambda_param: List[float],
        dict_fcns: Union[
            List[MultiVariableFunction], List[List[MultiVariableFunction]]
        ],
        state_name: Union[str, List[str]],
        config: Dict,
        mode: str,
    ):
        self.batch_mode = mode
        self.SBL_problems = []
        self.valid_solutions = []
        if self.batch_mode == "state_batch":
            if not isinstance(dict_mtx, list):
                dict_mtx = [dict_mtx] * len(state_name)
            if not isinstance(dict_fcns[0], list):
                dict_fcns = [dict_fcns] * len(state_name)
            for (
                one_data_vec,
                one_state,
                one_lambda,
                one_dict_mtx,
                one_dict_fcn,
            ) in zip(data_vec, state_name, lambda_param, dict_mtx, dict_fcns):
                self.SBL_problems.append(
                    SBL(
                        dict_mtx=one_dict_mtx,
                        data_vec=one_data_vec,
                        lambda_param=one_lambda,
                        dict_fcns=one_dict_fcn,
                        state_name=one_state,
                        config=config,
                    )
                )

        elif self.batch_mode == "lambda_sweep":
            if not isinstance(state_name, str):
                raise TypeError(
                    f"state_name mustbe a string, we got {type(state_name)}"
                )

            for one_lambda in lambda_param:
                self.SBL_problems.append(
                    SBL(
                        dict_mtx=dict_mtx,
                        data_vec=data_vec,
                        lambda_param=one_lambda,
                        dict_fcns=dict_fcns,
                        state_name=state_name,
                        config=config,
                    )
                )
        else:
            raise ValueError(f"{mode} is not a supported batch mode!")

    def compute_model_structure(self, max_iter=10):
        for SBL_problem in self.SBL_problems:
            try:
                with Timer(
                    name=f"{SBL_problem.state_name} with {max_iter} iter"
                ):
                    SBL_problem.compute_model_structure(max_iter=max_iter)
            except SBLError as e:
                warnings.warn(str(e))
                self.valid_solutions.append(False)
            else:
                self.valid_solutions.append(True)

    def get_results(self, zero_th):
        if not all(self.valid_solutions):
            raise SBLError("invalid SBL solution was found!")
        if self.batch_mode == "state_batch":
            ret_dict = {}
            for SBL_problem in self.SBL_problems:
                ret_dict.update(
                    {
                        SBL_problem.state_name: SBL_problem.get_results(
                            zero_th=zero_th
                        )
                    }
                )
            return ret_dict
        else:
            return [
                SBL_problem.get_results(zero_th=zero_th)
                for SBL_problem in self.SBL_problems
            ]


class ConvergenceMonitor(object):
    """A convergence monitor that calculates the number of dictionary indices that has converged to a
    stationary value based on the gamma estimates in SBL_problem.

    The actual convergence test is done by augmented Dickey–Fuller test
    """

    def __init__(
        self,
        SBL_problem,
        min_iter=5,
        perc_column_conv=0.95,
        mode="MOV_AVG",
        mov_avg_th=1e-8,
    ):
        """Initialize a Convergence Monitor instance

        Args:
            SBL_problem: instance of an SBL class
            min_iter: minimum number of iterations before convergence is checked
            perc_column_conv: percentage of columns that are converged
        """
        self.min_iter = min_iter
        self.SBL_problem = SBL_problem
        self.perc_column_conv = perc_column_conv
        self.converged = False
        self.mode = mode
        self.mov_avg_th = mov_avg_th
        self.p_value = 0.05

    def calculate_convergence(self):
        all_time_series = self.SBL_problem.gamma_estimates
        column_number = len(self.SBL_problem.dict_fcns)
        iter_num = len(all_time_series)

        if iter_num >= self.min_iter:
            status = []
            for column_idx in range(0, column_number):
                # extract the column_idx item from each list
                gamma_values = list(np.array(all_time_series).T[column_idx])
                if self.mode == "MOV_AVG":
                    status.append(self._moving_avg(gamma_values))
                elif self.mode == "ADF":
                    status.append(self._adfuller_mode(gamma_values))
                else:
                    raise ValueError(f"{self.mode} is not a valid mode!")

            converged = status.count(True)
            logger.debug(
                f"in {self.SBL_problem.state_name} so far {converged} out of {column_number} has converged after {iter_num} iterations"
            )

            if converged >= column_number * 0.9:
                self.converged = True

    def _adfuller_mode(self, gamma_values):
        adfuller_result = adfuller(gamma_values)
        # get the probability that null hypothesis will not be rejected
        status = adfuller_result[1]
        if status < self.p_value:
            return True
        else:
            return False

    def _moving_avg(self, gamma_values):
        return (
            np.mean(gamma_values[-(self.min_iter - 1) :]) - gamma_values[-1]
            < self.mov_avg_th
        )

    def is_converged(self):
        return self.converged


class RefitModel(object):
    def __init__(self, batch_SBL, dictionary, state_name):
        self.state_name = state_name
        self.batch_SBL = batch_SBL
        self.dictionary = dictionary

    def refit(self, orig_data, data_vec, config, zero_th):
        # set the lambda to zero, pure data fit, no model selection
        lambda_param = [0] * len(self.state_name)
        all_selected_dict_fcns = []
        dict_mtx = []

        # select the non zero RHS indices and build a new dictionary for each state
        for sbl in self.batch_SBL.SBL_problems:
            selected_dict_fcns = sbl.get_results(zero_th=zero_th)
            all_selected_dict_fcns.append(selected_dict_fcns)
            sub_dict = DictionaryBuilder.from_dict_fcns(selected_dict_fcns)
            A = sub_dict.evaluate_dict(input_data=orig_data)
            dict_mtx.append(A)

        new_sbls = BatchSBL(
            dict_mtx=dict_mtx,
            data_vec=data_vec,
            lambda_param=lambda_param,
            dict_fcns=all_selected_dict_fcns,
            state_name=self.state_name,
            config=config,
            mode="state_batch",
        )

        new_sbls.compute_model_structure(max_iter=1)

        return new_sbls
