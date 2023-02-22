import numpy as np


def three_hump_camel(x: np.ndarray) -> np.ndarray:
    """
    The Three-Hump Camel Function.
    Typically, evaluated on the input domain [-5, 5]^2.

    Dimensions: 2
    Global optimum: f(0,0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    - The value of the Three-Hump Camel Function at point x
    """
    x1, x2 = x
    return 2*x1**2 - 1.05*x1**4 + x1**6/6 + x1*x2 + x2**2


def six_hump_camel(x: np.ndarray) -> np.ndarray:
    """
    The Six-Hump Camel Function.
    Typically, evaluated on the input domain [-5, 5]^2.

    Dimensions: 2
    Global optimum: f(0.0898,-0.7126) = f(-0.0898,0.7126) = -1.0316

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    - The value of the Six-Hump Camel Function at point x
    """
    x1, x2 = x
    return (4 - 2.1*x1**2 + x1**4/3)*x1**2 + x1*x2 + (-4 + 4*x2**2)*x2**2


def dixon_price(x: np.ndarray) -> np.ndarray:
    """
    The Dixon-Price Function.
    Typically, evaluated on the input domain [2^-30, 2^30-1]^d.

    Dimensions: d
    Global optimum: f(2^(-1^(2^i-2)/(2^i-1))) = 0, i=2,3,...,d

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    - The value of the Dixon-Price Function at point x
    """
    d = len(x)
    x1 = x[0]
    summation = np.sum((i+1)*(2*np.power(x[1:], 2) - x[:-1])**2 for i in range(d-1))
    return np.power(x1-1, 2) + summation


def rosenbrock(x: np.ndarray) -> np.ndarray:
    """
    The Rosenbrock Function.
    Typically, evaluated on the input domain [-5, 10]^d.

    Dimensions: d
    Global optimum: f(1,1,...,1) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    - The value of the Rosenbrock Function at point x
    """
    return np.sum(100*(x[1:]-x[:-1]**2)**2 + (1-x[:-1])**2)
