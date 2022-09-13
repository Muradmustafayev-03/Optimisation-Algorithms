from imports import *


def perm0(x: Vector, beta=10) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-d, d], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (1, 1/2,...,1/d)
    """
    d = len(x)
    return sum(sum((j + beta) * (x[j] ** (i+1) - 1 / (j ** (i+1))) for j in range(d)) ** 2 for i in range(d))


def rotated_hyper_ellipsoid(x: Vector) -> float:
    """
        The function is usually evaluated on the hypercube xi ∈ [-65.536, 65.536], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (0,...,0)
    """
    d = len(x)
    return sum(sum(x[j] ** 2 for j in range(i)) for i in range(d))


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
    return sum(abs(x[i]) ** (i+2) for i in range(d))