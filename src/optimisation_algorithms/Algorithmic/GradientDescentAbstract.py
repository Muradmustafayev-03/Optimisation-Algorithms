from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import warnings


class BaseGD(ABC):
    """
    A base abstract class for gradient descent algorithms.

    Methods:
    -------
    generate_random_sample() -> np.ndarray[float]:
        Generates a random initial sample of dimension d of values between self.rand_min and self.rand_max
    gradient(x: np.ndarray) -> np.ndarray:
        Computes the gradient of a function f at point x.
    fit(maximize: bool = False) -> Tuple[np.ndarray, float]:
        Abstract method that finds the minimum or maximum of a function f using batch gradient descent
        starting from a random point.
    fit_multiple(self, num_runs: int = 10, maximize: bool = False) -> Tuple[np.ndarray, float]:
        Perform multiple runs of the optimization routine and return the best result.
    _selection(**kwargs) -> np.ndarray:
        Abstract method that selects a subset of features to use in the optimization process.
    _termination_criteria(**kwargs) -> bool:
        Abstract method that determines whether the optimization process should terminate.

    Raises:
    ------
    NotImplementedError:
        If either of the abstract methods is not implemented in a subclass.
    """

    @abstractmethod
    def __init__(self, f: callable, d: int, h: float = 1e-8, rand_min: float = 0, rand_max: float = 1):
        self.f = f
        self.d = d
        self.h = h
        self.rand_min = rand_min
        self.rand_max = rand_max

    def generate_random_sample(self) -> np.ndarray[float]:
        """
        Generates a random initial sample of dimension d of values between self.rand_min and self.rand_max

        Returns:
        -------
        x : np.ndarray
            An array of randomized values.
        """
        rng = np.random.default_rng()
        x = rng.random(self.d)
        return x * (self.rand_max - self.rand_min) + self.rand_min

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of a function f at point x.

        Parameters:
        ----------
        x : numpy.ndarray
            An array representing the point at which to compute the gradient.
        Returns:
        -------
        numpy.ndarray:
            An array representing the gradient of the function self.f at point x.
        """

        identity = np.identity(len(x))
        gradient = np.array(
            [self.f(x + self.h * identity[i]) - self.f(x - self.h * identity[i]) for i in range(len(x))]
            ) / (2 * self.h)
        return gradient

    @abstractmethod
    def fit(self, maximize: bool = False) -> Tuple[np.ndarray, float]:
        """
        Finds the minimum or maximum of a function f using batch gradient descent starting from a random point.

        Parameters:
        ----------
        maximize : bool (default: False)
            If True, the method will find the maximum of the function. Otherwise, the default is False, and the method
            will find the minimum of the function.
        Returns:
        -------
        x : np.ndarray
            The parameter values at the minimum or maximum of the function.
        f(x) : float
            The value of the function at the minimum or maximum.
        Raises:
        ------
        RuntimeWarning:
            Gradient failed to converge within the maximum number of iterations.
        """

    def fit_multiple(self, num_runs: int = 10, maximize: bool = False) -> Tuple[np.ndarray, float]:
        """
        Perform multiple runs of the optimization routine and return the best result.

        Parameters:
        -----------
        num_runs : int (default: 1)
            The number of optimization runs to perform.
        maximize : bool (default: False)
            Whether to maximize or minimize the objective function.
        Returns:
        --------
        x : np.ndarray
            The parameter values at the minimum or maximum of the function.
        f(x) : float
            The value of the function at the minimum or maximum.
        """
        best_solution, min_val = None, np.inf
        for _ in range(num_runs):
            x, f_x = self.fit(maximize)
            if f_x < min_val:
                best_solution, min_val = x, f_x
        return best_solution, min_val

    @abstractmethod
    def _selection(self, **kwargs) -> np.ndarray:
        """
        Selects a subset of features to use in the optimization process.
        """

    @abstractmethod
    def _termination_criteria(self, **kwargs) -> bool:
        """
        Determines whether the optimization process should terminate.
        """


class BatchGD(BaseGD, ABC):
    """
    Abstract class to be inherited for Batch Gradient Descent.
    """

    @abstractmethod
    def __init__(self, d: int, tol: float = 1e-8):
        self.d = d
        self.tol = tol

    def _selection(self) -> np.ndarray:
        return np.arange(self.d)

    def _termination_criteria(self, **kwargs) -> bool:
        return np.abs(np.linalg.norm(kwargs['grad'])) < self.tol


class MiniBatchGD(BatchGD, ABC):
    """
    Abstract class to be inherited for Mini-Batch Gradient Descent.
    """

    @abstractmethod
    def __init__(self, d: int, batch_size: int, tol: float = 1e-8):
        self.d = d
        self.tol = tol
        self.batch_size = batch_size

    def _selection(self) -> np.ndarray:
        return np.random.choice(self.d, self.batch_size)

    def _termination_criteria(self, **kwargs) -> bool:
        return np.abs(kwargs['f_new'] - kwargs['f_old']) < self.tol


class StochasticGD(MiniBatchGD, ABC):
    """
    Abstract class to be inherited for Stochastic Gradient Descent.
    """

    @abstractmethod
    def __init__(self, d: int, tol: float = 1e-8):
        MiniBatchGD.__init__(self, d=d, batch_size=1, tol=tol)


class SimpleGD(BaseGD, ABC):
    @abstractmethod
    def __init__(self, f: callable, d: int, learning_rate: float = 0.1, max_iter: int = 10 ** 5,
                 tol: float = 1e-8, h: float = 1e-8, rand_min: float = 0, rand_max: float = 1):
        BaseGD.__init__(self, f, d, h, rand_min, rand_max)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, maximize: bool = False) -> Tuple[np.ndarray, float]:
        x = self.generate_random_sample()
        sign = 1 if maximize else -1
        f_old = self.f(x)
        for i in range(self.max_iter):
            indices = self._selection()
            grad = self.gradient(x[indices])
            x[indices] += sign * self.learning_rate * grad
            f_new = self.f(x)
            if self._termination_criteria(grad=grad, f_old=f_old, f_new=f_new):
                break
            f_old = f_new
        else:
            warnings.warn("Gradient failed to converge within the maximum number of iterations.")
        return x, self.f(x)


class ConjugateGD(BaseGD, ABC):
    @abstractmethod
    def __init__(self, f: callable, d: int, max_iter: int = 10 ** 5,
                 tol: float = 1e-8, h: float = 1e-8, rand_min: float = 0, rand_max: float = 1):
        BaseGD.__init__(self, f, d, h, rand_min, rand_max)
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, maximize: bool = False) -> Tuple[np.ndarray, float]:
        x = self.generate_random_sample()
        sign = 1 if maximize else -1
        r = -self.gradient(x)
        f_old = self.f(x)
        p = r
        for i in range(self.max_iter):
            Ap = self.gradient(p)
            alpha = np.dot(r, r) / np.dot(p, Ap)
            x += alpha * p
            r_new = sign * self.gradient(x)
            f_new = self.f(x)
            if self._termination_criteria(grad=r_new, f_old=f_old, f_new=f_new):
                break
            f_old = f_new
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new
        else:
            warnings.warn("Gradient failed to converge within the maximum number of iterations.")

        return x, self.f(x)


class ExponentiallyWeightedGD(BaseGD, ABC):
    @abstractmethod
    def __init__(self, f: callable, d: int, learning_rate: float = 0.1, alpha: float = 0.9, max_iter: int = 10 ** 5,
                 tol: float = 1e-8, h: float = 1e-8, rand_min: float = 0, rand_max: float = 1):
        BaseGD.__init__(self, f, d, h, rand_min, rand_max)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol

    def fit(self, maximize: bool = False) -> Tuple[np.ndarray, float]:
        x = self.generate_random_sample()
        sign = 1 if maximize else -1
        f_old = self.f(x)
        v = 0  # Initialize exponentially weighted moving average
        for i in range(self.max_iter):
            grad = self.gradient(x)
            v = self.alpha * v + (1 - self.alpha) * grad**2
            x += sign * self.learning_rate * grad / np.sqrt(v + 1e-8)  # Add 1e-8 to avoid division by zero
            f_new = self.f(x)
            if self._termination_criteria(grad=grad, f_old=f_old, f_new=f_new):
                break
            f_old = f_new
        else:
            warnings.warn("Gradient failed to converge within the maximum number of iterations.")
        return x, self.f(x)
