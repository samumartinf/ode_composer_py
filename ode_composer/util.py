from sympy import *
from typing import List, Dict
from dataclasses import dataclass
import numpy as np
from sympy.parsing.sympy_parser import parse_expr
import math
import time
import logging
import os

logger = logging.getLogger(f"ode_composer_{os.getpid()}")


from typing import List, Dict
from sympy import Symbol, Mul, parse_expr
from sympy.utilities.lambdify import lambdify
import numpy as np
import math

@dataclass
class MultiVariableFunction:
    arguments: List[Symbol]
    # TODO restrict this better, check sympy documentation
    fcn_pointer: object
    symbolic_expression: Mul
    constant_name: str = ""
    constant: float = 1.0
    """
    Represents a multi-variable function.

    Args:
        arguments (List[Symbol]): The list of symbols representing the arguments of the function.
        fcn_pointer (object): The function pointer to evaluate the function.
        symbolic_expression (Mul): The symbolic expression of the function.
        constant_name (str, optional): The name of the constant. Defaults to "".
        constant (float, optional): The value of the constant. Defaults to 1.0.
    """

    def __repr__(self):
        """
        Returns a string representation of the symbolic expression.

        Returns:
            str: The string representation of the symbolic expression.
        """
        return str(self.symbolic_expression)

    def evaluate_function(self, measurement_data: Dict):
        """
        Evaluates the function using the provided measurement data.

        Args:
            measurement_data (Dict): A dictionary containing the measurement data.

        Returns:
            float: The result of evaluating the function.
        
        Raises:
            KeyError: If any of the arguments are missing in the measurement data.
        """
        data = list()
        for key in self.arguments:
            key = str(key)
            if key not in measurement_data.keys():
                raise KeyError(
                    "Missing data for %s in expression %s"
                    % (key, self.symbolic_expression)
                )
            data.append(np.array(measurement_data.get(key)))
        return self.fcn_pointer(*data)

    @staticmethod
    def create_function(
        rhs_fcn: str, parameters: Dict[str, float], weight: float
    ):
        """
        Creates a MultiVariableFunction object from the given right-hand side function, parameters, and weight.

        Args:
            rhs_fcn (str): The right-hand side function as a string.
            parameters (Dict[str, float]): A dictionary containing the parameter values.
            weight (float): The weight of the function.

        Returns:
            MultiVariableFunction: The created MultiVariableFunction object.
        """
        if "^" in rhs_fcn:
            rhs_fcn = rhs_fcn.replace("^", "**")
        sym_expr = parse_expr(s=rhs_fcn, evaluate=False, local_dict=parameters)
        expr_variables = list(sym_expr.free_symbols)
        func = lambdify(args=expr_variables, expr=sym_expr, modules="numpy")
        return MultiVariableFunction(
            arguments=expr_variables,
            fcn_pointer=func,
            symbolic_expression=sym_expr,
            constant=weight,
        )

    def get_constant_sign(self) -> str:
        """
        Returns the sign of the constant as a string.

        Returns:
            str: The sign of the constant. "+" if positive, "-" if negative.
        """
        sign = math.copysign(1, self.constant)
        if sign >= 0:
            return "+"
        else:
            return "-"


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            logger.info(f"{self.name} took {time.time() - self.tstart} sec")


def validate_data(data, data_name=None):
    if np.isinf(data).any() or np.isnan(data).any():
        raise ValueError(f"{data_name} contains invalid values!")
