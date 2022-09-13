from imports import *


def rosenbrock(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-5, 10], for all i = 1, …, d, \n,
        although it may be restricted to the hypercube xi ∈ [-2.048, 2.048], for all i = 1, …, d \n
        Global Minimum: f(x*) = 0 at x* = (1,...,1)
    """
    d = len(x)
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(d - 1))
