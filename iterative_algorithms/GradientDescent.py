import random
import numpy as np


def grad(func, x, delta=10 ** (-20)):
    def replace(array, init, place):
        new = np.copy(array)
        new[new == init] = place
        return new

    return np.array([(func(replace(x, x_i, x_i + delta)) - func(x)) / delta for x_i in x])


def batch_gradient_descent(gradient: callable, d: int, alpha: float = 0.2, tol: float = 10 ** (-20),
                           randomize: bool = False, max_iterations: int = 1000000, _range: int = 1000):
    try:
        current = (np.random.rand(d) - 0.5) * _range

        for _ in range(max_iterations):
            step = alpha * gradient(current) * (random.random() ** int(randomize))

            if step.all() < tol:
                break
            current = current - step

        return current

    except RuntimeWarning:
        return batch_gradient_descent(gradient, d, alpha, tol, randomize, max_iterations, _range)


def approximated_gradient_descent(func: callable, d: int, alpha: float = 0.2, tol: float = 10 ** (-20),
                                  grad_delta: float = 10 ** (-8), randomize: bool = False,
                                  max_iterations: int = 1000000, _range: int = 1000):
    current = (np.random.rand(d) - 0.5) * _range

    for _ in range(max_iterations):
        step = alpha * grad(func, current, grad_delta) * (random.random() ** int(randomize))

        if step.all() < tol:
            break
        current = current - step

    return current
