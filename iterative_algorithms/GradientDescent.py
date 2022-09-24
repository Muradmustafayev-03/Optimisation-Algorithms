import random
from abc import abstractmethod

import numpy as np


class GradientDescent:
    def __init__(self, d: int, _range: float = 1000, randomize: bool = False):
        self.d = d
        self.range = _range
        self.randomize = randomize

    @abstractmethod
    def grad(self, x: np.array) -> np.array:
        pass

    def optimize(self, max_iterations, alpha: float = 0.2, tol: float = 10 ** (-20)) -> np.array:
        current = (np.random.rand(self.d) - 0.5) * self.range

        for _ in range(max_iterations):
            step = self.grad(current) * alpha * (random.random() ** int(self.randomize))

            if step.all() < tol:
                break
            current = current - step

        return current


class BatchGradientDescent(GradientDescent):
    def __init__(self, gradient: callable, d: int, _range: float = 1000, randomize: bool = False):
        super().__init__(d, _range, randomize)
        self.gradient = gradient

    def grad(self, x: np.array) -> np.array:
        return self.gradient(x)


class ApproximatedGradientDescent(GradientDescent):
    def __init__(self, func: callable, d: int, _range: float = 1000, randomize: bool = False):
        super().__init__(d, _range, randomize)
        self.func = func

    @staticmethod
    def replace(array, init, place):
        new = np.copy(array)
        new[new == init] = place
        return new

    def grad(self, x: np.array, delta: float = 10 ** (-20)) -> np.array:
        return np.array([(self.func(self.replace(x, x_i, x_i + delta)) - self.func(x)) / delta for x_i in x])


#   ######################TO#DELETE#ALL#BELOW###############################

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
