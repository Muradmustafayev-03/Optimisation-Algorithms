import numpy as np


def booth(x: np.ndarray) -> float:
    """
    The Booth function.
    The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.

    Global optimum: f(1, 3) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Booth function at point x
    """
    x1, x2 = x
    return (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2


def matyas(x: np.ndarray) -> float:
    """
    The Matyas function.
    The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.

    Global optimum: f(0, 0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Matyas function at point x
    """
    x1, x2 = x
    return 0.26 * (x1**2 + x2**2) - 0.48*x1*x2


def mccormick(x: np.ndarray) -> float:
    """
    The McCormick function.
    The function is usually evaluated on the rectangle x1 ∈ [-1.5, 4], x2 ∈ [-3, 4].

    Global optimum: f(-0.54719, -1.54719) = -1.9133

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the McCormick function at point x
    """
    x1, x2 = x
    return np.sin(x1 + x2) + (x1 - x2)**2 - 1.5*x1 + 2.5*x2 + 1


def power_sum(X, a):
    """
    Power Sum Function
    The function is usually evaluated on the hypercube xi ∈ [0, d], for all i = 1, …, d.

    Parameters
    ----------
    X : numpy.ndarray
        Input array.
    a : float
        A parameter of the function.

    Returns
    -------
    float
        The value of the Power Sum function at the given input.

    """
    return np.sum(np.power(np.abs(X), a))


def zakharov(X):
    """
    Zakharov Function
    The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d.

    Parameters
    ----------
    X : numpy.ndarray
        Input array.

    Returns
    -------
    float
        The value of the Zakharov function at the given input.

    """
    n = len(X)
    sum_sq = np.sum(np.power(X, 2))
    sum_cos = np.sum(0.5 * np.multiply(np.arange(1, n+1), X))
    return sum_sq + np.power(sum_cos, 2) + np.power(sum_cos, 4)
