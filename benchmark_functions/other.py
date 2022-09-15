from .imports import *


def dixon_price(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-10, 10], for all i = 1,..., d \n
         Global Minimum: f(x*) = 0 at x_i = 2^(-(2^i - 2)/2^i) for i = 1,...,d
    """
    d = len(x)
    return (x[0] - 1) ** 2 + sum((i + 1) * (2 * x[i] ** 2 - x[i - 1]) ** 2 for i in range(1, d))


def michalewicz(x: Vector, m=10) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [0, π], for all i = 1, ..., d \n
        Global Minima: \n
        at len(x) = 2: f(x*) = -1.8013 at x* = (2.20, 1.57) \n
        at len(x) = 5: f(x*) = -4.687658 \n
        at len(x) = 2: f(x*) = -9.66015 \n
    """
    d = len(x)
    return -sum(sin(x[i] * (sin((i + 1) * x[i] ** 2 / pi)) ** (2 * m)) for i in range(d))


def perm(x: Vector, beta=10) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-d, d], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (1,2,...,d)
    """
    d = len(x)
    return sum(sum(((j+1) ** (i+1) + beta) * ((x[j] / (j+1)) ** (i+1) - 1) for j in range(d)) ** 2 for i in range(d))


def rosenbrock(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1,..., d, \n,
        although it may be restricted to the hypercube xi ∈ [-2.048, 2.048], for all i = 1,..., d \n
        Global Minimum: f(x*) = 0 at x* = (1,...,1)
    """
    d = len(x)
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(d - 1))


def styblinski_tang(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-5, 5], for all i = 1, ..., d \n
        Global Minimum: f(x*) =  -39.16599d at x* = (2.903534,...,2.903534)
    """
    return sum(x_i ** 4 - 16 * x_i ** 2 + 5 * x_i for x_i in x) / 2


def zakharov(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1,..., d \n
        Global Minimum: f(x*) = 0 at x* = (0,...,0)
    """
    d = len(x)
    return sum(x[i] ** 2 for i in range(d)) + \
           sum(0.5 * (i + 1) * x[i] for i in range(d)) ** 2 + \
           sum(0.5 * (i + 1) * x[i] for i in range(d)) ** 4
