import numpy as np


def perm_function(x: np.ndarray, beta: float) -> float:
    """
    The Perm Function.
    Typically, evaluated on the input domain [0, 1]^d.

    Dimensions: d
    Global optimum: f(x_1, x_2, ..., x_d) = 0, with x_i = i/d for i = 1,...,d

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function
    - beta: a scalar parameter

    Returns:
    - The value of the Perm Function at point x with parameter beta
    """
    d = len(x)
    idx = np.arange(d) + 1
    return np.power(np.abs(np.power(idx, beta) / idx - x)).sum()


def power_sum(x: np.ndarray, b: float = 2.0) -> float:
    """
    The Power Sum Function.
    Typically, evaluated on the input domain [-1, 1]^d.

    Dimensions: d
    Global optimum: f(0,...,0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function
    - b: a float value that controls the steepness of the valleys (default is 2.0)

    Returns:
    - The value of the Power Sum Function at point x
    """
    return np.sum(np.power(np.abs(x), b)) + np.power(np.sum(np.abs(x)), b)
