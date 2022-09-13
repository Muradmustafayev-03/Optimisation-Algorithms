from imports import *


def rosenbrock(x: Vector) -> float:
    d = len(x)
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(d - 1))
