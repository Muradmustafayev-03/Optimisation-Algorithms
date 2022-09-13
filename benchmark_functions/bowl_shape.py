from imports import *


def perm0(x: Vector, beta=10):
    """
        The function is usually evaluated on the hypercube xi ∈ [-d, d], for all i = 1, ..., d \n
        Global Minimum: f(x*) = 0 at x* = (1, 1/2,...,1/d)
    """
    d = len(x)
    return sum(sum((j + beta) * (x[j] ** (i+1) - 1 / (j ** (i+1))) for j in range(d)) ** 2 for i in range(d))
