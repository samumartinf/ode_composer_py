from ode_composer.statespace_model import StateSpaceModel
from scipy.integrate import solve_ivp
from typing import List, Dict, Union
import numpy as np


class MeasurementsGenerator(object):
    """
    A class that provides functionality to generate measurements from a StateSpaceModel.

    Methods:
        get_measurements: Generates measurements based on the solution of the ODE system.

    Attributes:
        ss (StateSpaceModel): The state space model representing the ODE system.
        sol (scipy.integrate.OdeSolution): The solution of the ODE system.
    """

    def __init__(
        self,
        ss: StateSpaceModel,
        time_span: List[float],
        initial_values: Dict[str, float],
        data_points: int = None,
    ):
        """
        Initializes a MeasurementsGenerator object.

        Args:
            ss (StateSpaceModel): The state space model representing the ODE system.
            time_span (List[float]): The time span over which to solve the ODE system.
            initial_values (Dict[str, float]): The initial values of the state variables.
            data_points (int, optional): The number of data points to generate. If not provided,
                the time points will be obtained from the solution of the ODE system.

        Raises:
            ValueError: If the integration fails.
        """
        self.ss = ss
        states = initial_values.keys()
        if data_points is None:
            t_eval = None
        else:
            t_eval = np.linspace(time_span[0], time_span[1], data_points)
        sol = solve_ivp(
            fun=self.ss.get_rhs,
            t_span=time_span,
            y0=list(initial_values.values()),
            args=(states,),
            t_eval=t_eval,
        )

        if sol.success is not True:
            raise ValueError(f"Integration Problem {sol.message}")

        self.sol = sol

    def get_measurements(self, SNR_db=None):
        """
        Generates measurements based on the solution of the ODE system.

        Args:
            SNR_db (float, optional): Signal-to-Noise Ratio in decibels. If provided, the measurements
                will be corrupted with additive white Gaussian noise.

        Returns:
            tuple: A tuple containing the time points and the generated measurements.
                - The time points are obtained from the solution of the ODE system.
                - The measurements are obtained by adding noise to the solution, if SNR_db is provided.
        """
        if SNR_db is not None:
            y_measured = np.zeros(shape=self.sol.y.shape)
            SNR = 10 ** (SNR_db / 10)
            for idx, y in enumerate(self.sol.y):
                Esym = np.sum(abs(y) ** 2) / len(y)
                N_PSD = (Esym) / SNR
                y_measured[idx, :] = y + np.sqrt(N_PSD) * np.random.randn(
                     len(y)
                 )
                
        else:
            y_measured = self.sol.y

        return self.sol.t, y_measured
