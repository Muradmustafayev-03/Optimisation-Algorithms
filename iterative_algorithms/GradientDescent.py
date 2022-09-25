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

    def optimize(self, max_iterations: int = 10000, alpha: float = 0.2, tol: float = 10 ** (-20)) -> np.array:
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

    def grad(self, x: np.array, delta: float = 10 ** (-10)) -> np.array:
        return np.array([(self.func(self.replace(x, x_i, x_i + delta)) - self.func(x)) / delta for x_i in x])
