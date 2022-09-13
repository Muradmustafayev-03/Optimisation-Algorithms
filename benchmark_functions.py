from numpy import exp, square, sqrt, sin, cos, pi
import numpy as np

Vector = list[float]


def ackley(x: Vector, a=20, b=0.2, c=2 * pi) -> int:
    d = len(x)
    return -1 * a * exp(-1 * b * sqrt((1 / d) * np.sum(square(x)))) - exp(
        (1 / d) * np.sum(cos(x * np.full(shape=d, fill_value=c)))) + a + exp(1)


def levy(x: Vector) -> int:
    d = len(x)
    w = [1 + (i - 1) / 4 for i in x]
    return sin(pi * w[0]) ** 2 + \
           sum((w[i] - 1) ** 2 * (1 + 10 * sin(pi * w[i] + 1) ** 2) for i in range(d - 1)) + \
           (w[d-1] - 1) ** 2 * (1 + 10 * sin(2 * pi * w[d-1] + 1) ** 2)


def rastrigin(x: Vector) -> int:
    d = len(x)
    return 10 * d + sum(x[i] ** 2 - 10 * cos(2 * pi * x[i]) for i in range(d))


def rosenbrock(x: Vector) -> int:
    d = len(x)
    return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (x[i] - 1) ** 2 for i in range(d - 1))


def schwefel(x: Vector) -> int:
    d = len(x)
    return 418.9829 * d - sum(x[i] * sin(sqrt(abs(x[i]))) for i in range(d))


def shubert(x: Vector) -> int:
    return sum(i * cos(i + 1) * x[0] for i in range(1, 6)) * sum(i * cos(i + 1) * x[1] for i in range(1, 6))
