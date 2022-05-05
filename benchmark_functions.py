from numpy import exp, square, sqrt, sin, cos, pi
import numpy as np


def ackley(x: list, a=20, b=0.2, c=2 * pi):
    d = len(x)
    return -1 * a * exp(-1 * b * sqrt((1 / d) * np.sum(square(x)))) - exp(
        (1 / d) * np.sum(cos(x * np.full(shape=d, fill_value=c)))) + a + exp(1)


def rastrigin(x: list):
    d = len(x)
    return 10 * d + sum(x[i] ** 2 - 10 * cos(2 * pi * x[i]) for i in range(d))


def rosenbrock(x: list):
    d = len(x)
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(d - 1))


def schwefel(x: list):
    d = len(x)
    return 418.9829 * d - sum(x[i] * sin(sqrt(abs(x[i]))) for i in range(d))
