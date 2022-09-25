from .imports import *


def ackley(x: Vector, a=20, b=0.2, c=2*pi) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-32.768, 32.768], for all i = 1, ..., d \n
        Recommended variable values are: a = 20, b = 0.2 and c = 2π \n
        Global Minimum: f(x*) = 0 at x* = (0,...,0)
    """
    d = len(x)
    return -1 * a * exp(-1 * b * sqrt((1 / d) * np.sum(square(x)))) - exp(
        (1 / d) * np.sum(cos(x * np.full(shape=d, fill_value=c)))) + a + exp(1)


def griewank(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-600, 600], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (0,...,0)
    """
    d = len(x)
    return sum(x[i] ** 2 / 4000 for i in range(d)) - prod([cos(x[i] / sqrt(i+1)) for i in range(d)]) + 1


def levy(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-10, 10], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (1,...,1)
    """
    d = len(x)
    w = [1 + (i - 1) / 4 for i in x]
    return sin(pi * w[0]) ** 2 + \
           sum((w[i] - 1) ** 2 * (1 + 10 * sin(pi * w[i] + 1) ** 2) for i in range(d - 1)) + \
           (w[d-1] - 1) ** 2 * (1 + 10 * sin(2 * pi * w[d-1] + 1) ** 2)


def rastrigin(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (0,...,0)
    """
    d = len(x)
    return 10 * d + sum(x[i] ** 2 - 10 * cos(2 * pi * x[i]) for i in range(d))


def schwefel(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-500, 500], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (420.9687,...,420.9687)
    """
    d = len(x)
    return 418.9829 * d - sum(x[i] * sin(sqrt(abs(x[i]))) for i in range(d))
