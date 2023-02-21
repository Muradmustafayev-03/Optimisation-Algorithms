import numpy as np


def bohachevsky(x: np.ndarray) -> float:
    """
    The Bohachevsky function.
    Typically, evaluated on the input domain [-100, 100] x [-100, 100].

    Dimensions: 2
    Global optimum: f(0, 0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Bohachevsky function at point x
    """
    x1, x2 = x
    return x1 ** 2 + 2 * x2 ** 2 - 0.3 * np.cos(3 * np.pi * x1) - 0.4 * np.cos(4 * np.pi * x2) + 0.7


def perm0(x: np.ndarray, beta: float) -> np.ndarray:
    """
    The Perm Function 0.
    Typically, evaluated on the input domain [-1, 1]^d.

    Dimensions: d
    Global optimum: f(0,...,d-1) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function
    - beta: a float parameter controlling the "sharpness" of the function

    Returns:
    -------
    - The value of the Perm Function 0 at point x
    """
    d = len(x)
    p = np.arange(1, d + 1)
    inner_sum = (np.power(np.abs(x), p) + beta) / p
    return np.sum(np.power(inner_sum, 10))


def rotated_hyper_ellipsoid(x: np.ndarray) -> np.ndarray:
    """
    The Rotated Hyper-Ellipsoid Function.
    Typically, evaluated on the input domain [-65.536, 65.536]^d.

    Dimensions: d
    Global optimum: f(0,...,0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    - The value of the Rotated Hyper-Ellipsoid Function at point x
    """
    d = len(x)
    return np.sum(np.power(np.dot(np.tril(np.ones((d, d))), x), 2))


def sphere(x: np.ndarray) -> np.ndarray:
    """
    The Sphere Function.
    Typically, evaluated on the input domain [-5.12, 5.12]^d.

    Dimensions: d
    Global optimum: f(0,...,0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Sphere Function at point x
    """
    return np.sum(np.power(x, 2))


def sum_of_different_powers(x: np.ndarray) -> np.ndarray:
    """
    The Sum of Different Powers Function.
    Typically, evaluated on the input domain [-1, 1]^d.

    Dimensions: d
    Global optimum: f(0,...,0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Sum of Different Powers Function at point x
    """
    d = len(x)
    powers = np.arange(1, d + 1)
    return np.sum(np.power(np.abs(x), powers))


def sum_squares(x: np.ndarray) -> np.ndarray:
    """
    The Sum Squares Function.
    Typically, evaluated on the input domain [-10, 10]^d.

    Dimensions: d
    Global optimum: f(0,...,0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Sum Squares Function at point x
    """
    d = len(x)
    return np.sum(np.arange(1, d+1) * np.power(x, 2))


def trid(x: np.ndarray) -> np.ndarray:
    """
    The Trid Function.
    Typically, evaluated on the input domain [-d^2, d^2]^d.

    Dimensions: d
    Global optimum: f(0,...,0) = -d(d+4)/6

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Trid Function at point x
    """
    return np.sum(np.power(x - 1, 2)) - np.sum(x[1:] * x[:-1])
