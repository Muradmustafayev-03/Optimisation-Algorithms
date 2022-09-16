from .imports import *


# seem not to work correctly
def rosenbrock_gradient(x: Vector):
    d = len(x)
    return np.array([400 * (x[i+1] - x[i] ** 2) * x[i] for i in range(d-1)] + [400 * (- x[-1]) ** 2 * x[-1]])


def rotated_hyper_ellipsoid_gradient(x: Vector):
    pass


def sphere_gradient(x: Vector):
    return np.array([2 * x_i for x_i in x])


def sum_of_powers_gradient(x: Vector):
    d = len(x)
    return np.array([(i+2) * abs(x[i]) ** (i+1) for i in range(d)])


def sum_of_squares_gradient(x: Vector):
    pass


def trid(x: Vector):
    pass
