import random
from abc import abstractmethod
import numpy as np
from .exceptions.FailedToConverge import FailedToConverge


class GradientDescent:
    """
    Optimisation using Gradient Descent.

    Attributes
    ----------
    d: int
        Number of dimensions of the function [f(x) = f(x_1, ..., x_d)]
    _range: float
        Range of values for the initial population x_1 in (-range/2; range/2)

    Methods
    -------
    grad(self, x: np.array) -> numpy.array
        Abstract method, to be overridden in child classes to return gradient of the function.

    optimize(self, max_iterations: int = 10000, alpha: float = 0.2, tol: float = 10 ** (-20),
                 randomize: bool = False) -> numpy.array
        Finds a local minima of the function.
    :return:
    """

    def __init__(self, d: int, _range: float = 1000):
        """
        Parameters
        ----------
        d: int
            Number of dimensions of the function [f(x) = f(x_1, ..., x_d)]
        _range: float
            Range of values for the initial population x_1 in (-range/2; range/2)

        :return: None
        """
        self.d = d
        self.range = _range

    @abstractmethod
    def grad(self, x: np.array) -> np.array:
        pass

    def optimize(self, max_iterations: int = 100000, alpha: float = 0.02, tol: float = 10 ** (-20),
                 randomize: bool = False) -> np.array:
        """
        Finds a local minima of the function.

        Parameters
        ----------
        max_iterations: int
            Maximal number of iterations
        alpha: float
            Descent change rate
        tol: float
            Tolerance number, maximal number to be considered as 0
        randomize: bool
            If True, randomizes change rate each step

        Returns
        -------
        :return: numpy.array
            x_min = (x_1, ..., x_d) where f(x_min) = min f(x)
        """
        current = (np.random.rand(self.d) - 0.5) * self.range

        for _ in range(max_iterations):
            try:
                step = self.grad(current) * alpha * (random.random() ** int(randomize))

                if step.all() < tol:
                    break
                current = current - step
            except RuntimeWarning as w:
                raise FailedToConverge().with_traceback(w.__traceback__)

        return current


class BatchGradientDescent(GradientDescent):
    """
    Optimisation using Batch Gradient Descent.

    Attributes
    ----------
    gradient: callable
        Gradient of the function
    d: int
        Number of dimensions of the function [f(x) = f(x_1, ..., x_d)]
    _range: float
        Range of values for the initial population x_1 in (-range/2; range/2)

    Methods
    -------
    grad(self, x: np.array) -> numpy.array
        Gradient of the function.

    optimize(self, max_iterations: int = 10000, alpha: float = 0.2, tol: float = 10 ** (-20),
                 randomize: bool = False) -> numpy.array
        Inherited from the parent class. Finds a local minima of the function.
    :return:
    """

    def __init__(self, gradient: callable, d: int, _range: float = 1000):
        """
        Parameters
        ----------
        gradient: callable
            Gradient of the function
        d: int
            Number of dimensions of the function [f(x) = f(x_1, ..., x_d)]
        _range: float
            Range of values for the initial population x_1 in (-range/2; range/2)

        :return: None
        """
        super().__init__(d, _range)
        self.gradient = gradient

    def grad(self, x: np.array) -> np.array:
        return self.gradient(x)


class ApproximatedGradientDescent(GradientDescent):
    """
    Optimisation using Batch Gradient Descent.

    Attributes
    ----------
    function: callable
        Function to minimize
    d: int
        Number of dimensions of the function [f(x) = f(x_1, ..., x_d)]
    _range: float
        Range of values for the initial population x_1 in (-range/2; range/2)

    Methods
    -------
    replace(array, init, place)
        Replaces an element in a numpy.array

    grad(self, x: np.array) -> numpy.array
        Gradient of the function.

    optimize(self, max_iterations: int = 10000, alpha: float = 0.2, tol: float = 10 ** (-20),
                 randomize: bool = False) -> numpy.array
        Inherited from the parent class. Finds a local minima of the function.
    :return:
    """

    def __init__(self, func: callable, d: int, _range: float = 1000):
        """
        Parameters
        ----------
        function: callable
            Function to minimize
        d: int
            Number of dimensions of the function [f(x) = f(x_1, ..., x_d)]
        _range: float
            Range of values for the initial population x_1 in (-range/2; range/2)

        :return: None
        """
        super().__init__(d, _range)
        self.func = func

    @staticmethod
    def replace(array, init, place):
        new = np.copy(array)
        new[new == init] = place
        return new

    def grad(self, x: np.array, delta: float = 10 ** (-10)) -> np.array:
        return np.array([(self.func(self.replace(x, x_i, x_i + delta)) - self.func(x)) / delta for x_i in x])
