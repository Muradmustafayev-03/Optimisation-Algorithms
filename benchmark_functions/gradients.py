from .imports import *


def rotated_hyper_ellipsoid_gradient(x: Vector):
    d = len(x)
    return np.array([2 * (d-i) * x[i] for i in range(d)])


def sphere_gradient(x: Vector):
    return np.array([2 * x_i for x_i in x])


def sum_of_powers_gradient(x: Vector):
    d = len(x)
    return np.array([(i+2) * abs(x[i]) ** (i+1) if x[i] >= 0 else -(i+2) * abs(x[i]) ** (i+1) for i in range(d)])


def sum_of_squares_gradient(x: Vector):
    d = len(x)
    return np.array([2 * (i+1) * x[i] for i in range(d)])


def trid_gradient(x: Vector):
    d = len(x)
    if d == 0:
        return [0]
    if d == 1:
        return [1]
    return np.array([2 * (x[i] - 1) for i in range(d)]) - \
           np.array([x[1]] + [x[i-1] + x[i+1] for i in range(1, d-1)] + [x[-2]])
