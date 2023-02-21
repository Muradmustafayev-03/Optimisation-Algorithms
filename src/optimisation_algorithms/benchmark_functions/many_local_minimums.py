import numpy as np


def ackley(x: np.ndarray) -> float:
    """
    The Ackley Function.
    Typically, evaluated on the input domain [-32.768, 32.768]^d.

    Dimensions: d
    Global optimum: f(0,...,0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Ackley Function at point x
    """
    d = len(x)
    term1 = -0.2 * np.sqrt(np.sum(np.power(x, 2)) / d)
    term2 = np.sum(np.cos(2 * np.pi * x)) / d
    return -20 * np.exp(term1) - np.exp(term2) + 20 + np.exp(1)


def bukin(x: np.ndarray) -> float:
    """
    The Bukin Function N. 6.
    Typically, evaluated on the input domain [-15, -5] x [-3, 3].

    Dimensions: 2
    Global optimum: f(-10, 1) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Bukin Function N. 6 at point x
    """
    x1, x2 = x
    term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * np.power(x1, 2)))
    term2 = 0.01 * np.abs(x1 + 10)
    return term1 + term2


def cross_in_tray(x: np.ndarray) -> float:
    """
    The Cross-in-Tray Function.
    Typically, evaluated on the input domain [-10, 10]^2.

    Dimensions: 2
    Global optimum: f(1.3491, -1.3491) = -2.06261

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Cross-in-Tray Function at point x
    """
    x1, x2 = x
    term1 = np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi)
    term2 = np.abs(np.sin(x1) * np.sin(x2) * np.exp(term1))
    return -0.0001 * np.power(term2 + 1, 0.1)


def drop_wave(x: np.ndarray) -> float:
    """
    The Drop-Wave Function.
    Typically, evaluated on the input domain [-5.12, 5.12]^d.

    Dimensions: 2
    Global optimum: f(0,0) = -1

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Drop-Wave Function at point x
    """
    x1, x2 = x
    numerator = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    denominator = 0.5 * (x1**2 + x2**2) + 2
    return -numerator / denominator


def eggholder(x: np.ndarray) -> float:
    """
    The Eggholder function.
    Typically, evaluated on the input domain [-512, 512]^2.

    Dimensions: 2
    Global optimum: f(512, 404.2319) = -959.6407

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Eggholder function at point x
    """
    x1, x2 = x
    term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    return term1 + term2 - 959.6407


def gramacy_lee(x: float) -> float:
    """
    The Gramacy & Lee (2012) function.
    Typically, evaluated on the input domain [0, 1]^d.

    Dimensions: 1
    Global optimum: f(0.548563362974474, -0.550903198335186) = -0.869011135358703

    Arguments:
    ---------
    - x: a float representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Gramacy & Lee (2012) function at point x
    """
    term1 = np.sin(10 * np.pi * x) / (2 * x)
    term2 = (x - 1) ** 4
    return term1 + term2 - 0.5


def griewank(x: np.ndarray) -> float:
    """
    The Griewank function.
    Typically, evaluated on the input domain [-600, 600]^d.

    Dimensions: d
    Global optimum: f(0, ..., 0) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Griewank function at point x
    """
    d = len(x)
    term1 = np.sum(np.power(x, 2)) / 4000
    term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, d + 1))))
    return 1 + term1 - term2


def holder_table(x: np.ndarray) -> float:
    """
    The Holder Table function.
    Typically, evaluated on the input domain [-10, 10]^2.

    Dimensions: 2
    Global optimum: f(8.05502, 9.66459) = -19.2085

    Arguments:
    ---------
    - x: a NumPy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Holder Table function at point x
    """
    x1, x2 = x
    term1 = -np.abs(np.sin(x1) * np.cos(x2) * np.exp(np.abs(1 - np.sqrt(x1 ** 2 + x2 ** 2) / np.pi)))
    return term1 - 19.2085


def langermann(x: np.ndarray, A: np.ndarray = None, c: np.ndarray = None, W: np.ndarray = None) -> float:
    """
    The Langermann function.
    Typically, evaluated in the domain [0, 10]^d, where d is the number of input dimensions.

    Dimensions: d
    Global optimum: Unknown

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function
    - A: a NumPy array of shape (m, d) containing the m coefficient sets
    - c: a NumPy array of shape (m,) containing the m constant offsets
    - W: a NumPy array of shape (m, d) containing the m frequency sets

    Returns:
    -------
    - The value of the Langermann function at point x
    """
    if A is None:
        A = np.random.rand(5, x.shape[0])
    if c is None:
        c = np.random.rand(5)
    if W is None:
        W = np.random.rand(5, x.shape[0])

    inner_sum = np.sum(A * np.exp(-(1 / np.pi) * np.sum(np.square(x - W), axis=1)), axis=1)
    return -np.sum(c * inner_sum)


def levy(x: np.ndarray) -> float:
    """
    The Levy function.
    Typically, evaluated in the domain [-10, 10]^d, where d is the number of input dimensions.

    Dimensions: d
    Global optimum: f(1, 1, ..., 1) = 0

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Levy function at point x
    """
    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0])) ** 2
    term2 = np.sum((w[:-1] - 1) ** 2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1)) ** 2))
    term3 = (w[-1] - 1) ** 2 * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)
    return term1 + term2 + term3


def levy_n13(x: np.ndarray) -> np.ndarray:
    """
    The Levy Function N. 13.
    Typically, evaluated on the square xi ∈ [-10, 10], for all i = 1, 2.

    Dimensions: d
    Global optimum: f(1,...,1) = 0

    Arguments:
    ----------
    - x: a NumPy array of shape (n,) representing the point at which to evaluate the function

    Returns:
    --------
    - The value of the Levy Function N. 13 at point x
    """

    w = 1 + (x - 1) / 4
    term1 = (np.sin(np.pi * w[0])) ** 2
    term2 = ((w[:-1] - 1) ** 2) * (1 + 10 * (np.sin(np.pi * w[:-1] + 1) ** 2))
    term3 = ((w[-1] - 1) ** 2) * (1 + (np.sin(2 * np.pi * w[-1])) ** 2)
    return np.sum(term1 + np.sum(term2) + term3)


def rastrigin(x: np.ndarray) -> float:
    """
    The Rastrigin Function.
    Typically, evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, …, d.

    Dimensions: d
    Global optimum: f(0,...,0) = 0

    Arguments:
    ----------
    - x: a NumPy array of shape (n,) representing the point at which to evaluate the function

    Returns:
    --------
    - The value of the Rastrigin Function at point x
    """
    d = x.shape[0]
    return 10 * d + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))


def schaffer_n2(x: np.ndarray) -> float:
    """
    The Schaffer Function N. 2.

    Domain: [-100, 100]^2
    Dimensions: 2
    Global minimum: f(0,0) = 0

    Arguments:
    ----------
    - x: a numpy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    --------
    - The value of the Schaffer Function N. 2 at point x
    """
    x1, x2 = x
    numerator = np.square(np.sin(np.sqrt(x1 ** 2 + x2 ** 2))) - 0.5
    denominator = np.square(1 + 0.001 * (x1 ** 2 + x2 ** 2))
    return 0.5 + numerator / denominator


def schaffer_n4(x: np.ndarray) -> float:
    """
    The Schaffer Function N. 4.

    Domain: [-100, 100]^2
    Dimensions: 2
    Global minimum: f(0, ±1.25313) = 0.292579

    Arguments:
    ----------
    - x: a numpy array of shape (2,) representing the point at which to evaluate the function

    Returns:
    --------
    - The value of the Schaffer Function N. 4 at point x
    """
    x1, x2 = x
    term1 = np.cos(np.sin(np.abs(x1 ** 2 - x2 ** 2)))
    term2 = 1 + 0.001 * (x1 ** 2 + x2 ** 2)
    return 0.5 + (term1 ** 2 - 0.5) / (term2 ** 2)


def schwefel(x: np.ndarray) -> float:
    """
    The Schwefel Function.
    Typically, evaluated on the hypercube xi ∈ [-500, 500]^n

    Dimensions: d
    Global minimum: f(x*) = 0 at x* = (420.9687,..., 420.9687)

    Arguments:
    ----------
    - x: a numpy array of shape (n,) representing the point at which to evaluate the function

    Returns:
    --------
    - The value of the Schwefel Function at point x
    """
    return - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def shubert(x: np.ndarray) -> np.ndarray:
    """
    The Shubert Function.
    The function is usually evaluated on the square xi ∈ [-10, 10], for all i = 1, 2,
    although this may be restricted to the square xi ∈ [-5.12, 5.12], for all i = 1, 2.

    Dimensions: d
    Global optimum: f(x) = -186.7309 (multiple global optima)

    Arguments:
    ---------
    - x: a NumPy array of shape (d,) representing the point at which to evaluate the function

    Returns:
    -------
    - The value of the Shubert Function at point x
    """

    x1 = np.outer(x, np.arange(1, 6))
    x2 = np.outer(x, np.ones(5))
    return np.sum(np.sin(x1) * np.cos((x1 * 2 + x2) / 2), axis=1)
