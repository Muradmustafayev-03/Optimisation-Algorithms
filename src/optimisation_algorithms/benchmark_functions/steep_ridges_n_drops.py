import numpy as np


def dejong5(x: np.ndarray) -> float:
    """
    De Jong Function N. 5.
    Typically, evaluated on the input domain [-65.536, 65.536]^2.

    Dimensions: 2
    Global optimum: f(1.584, 1.584) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    - The value of the De Jong Function N. 5 at point x
    """
    return np.sum(np.power(np.abs(np.square(x) - np.roll(np.square(x), -1)), 4)) + np.sum(np.power(x, 2))


def easom(x: np.ndarray) -> float:
    """
    The Easom Function.
    Typically, evaluated on the input domain [-100, 100]^2.

    Dimensions: 2
    Global optimum: f(pi, pi) = -1

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    - The value of the Easom Function at point x
    """
    return -np.cos(x[0]) * np.cos(x[1]) * np.exp(-np.square(x[0] - np.pi) - np.square(x[1] - np.pi))


def michalewicz(x: np.ndarray, m: int = 10) -> float:
    """
    The Michalewicz Function.
    Typically, evaluated on the input domain [0, pi]^d.

    Dimensions: d
    Global optimum: unknown

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function
    - m: a positive integer parameter

    Returns:
    - The value of the Michalewicz Function at point x
    """
    d = len(x)
    i = np.arange(1, d + 1)
    return -np.sum(np.sin(x) * np.power(np.sin(i * np.square(x) / np.pi), 2 * m))
