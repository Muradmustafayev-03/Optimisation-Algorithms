from .imports import *


def perm0(x: Vector, beta=10) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-d, d], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (1, 1/2,...,1/d)
    """
    d = len(x)
    return sum(sum((j + 1 + beta) * (x[j] ** (i + 1) - 1 / ((j + 1) ** (i + 1)))
                   for j in range(d)) ** 2 for i in range(d))


def rotated_hyper_ellipsoid(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-65.536, 65.536], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (0,...,0)
    """
    d = len(x)
    return sum((d-i) * (x[i] ** 2) for i in range(d))


def sphere(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-5.12, 5.12], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (0,...,0)
    """
    return sum(x_i ** 2 for x_i in x)


def sum_of_powers(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-1, 1], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (0,...,0)
    """
    d = len(x)
    return sum(abs(x[i]) ** (i + 2) for i in range(d))


def sum_of_squares(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-10, 10], for all i = 1, ..., d, \n
        although this may be restricted to the hypercube xi ∈ [-5.12, 5.12], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (0,...,0)
    """
    d = len(x)
    return sum((i + 1) * x[i] ** 2 for i in range(d))


def trid(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-d2, d2], for all i = 1, ..., d \n
        Global Minimum: f(x*) = -d(d+4)(d-1)/6, at x_i = i(d+1-i), for all i in 1,2,...,d
    """
    d = len(x)
    return sum((x[i] - 1) ** 2 for i in range(d)) - sum(x[i] * x[i - 1] for i in range(1, d))
