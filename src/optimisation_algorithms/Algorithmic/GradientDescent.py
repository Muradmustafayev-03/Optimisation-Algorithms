import warnings
from typing import Tuple
import numpy as np


class GradientDescent:
    """
    Batch Gradient Descent.
    Attributes:
    ----------
    learning_rate : float
        The learning rate to use in the gradient descent update.
    tol : float
        The convergence threshold for the norm of the gradient.
    max_iter : int
        The maximum number of iterations to run before stopping.
    Methods:
    -------
    gradient(f: callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        Computes the gradient of a function f at point x.
    fit(self, f: callable, d: int, maximize: bool = False) -> Tuple[np.ndarray, float]:
        Finds the minimum or maximum of a function f using gradient descent starting from a random point.
    """

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 100000, tol: float = 1e-8,
                 rand_min: float = 0, rand_max: float = 1):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.rand_min = rand_min
        self.rand_max = rand_max

    @staticmethod
    def gradient(f: callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """
        Computes the gradient of a function f at point x.
        Parameters:
        ----------
        f : callable
            A callable function.
        x : numpy.ndarray
            An array representing the point at which to compute the gradient.
        h : float
            A small value to use in the computation of the gradient.
        Returns:
        -------
        numpy.ndarray:
            An array representing the gradient of f at point x.
        """

        assert callable(f), "f must be a callable function"
        assert isinstance(h, float) and h > 0, "h should be a positive float"
        assert isinstance(x, np.ndarray), "x should be a NumPy array"

        gradient = np.zeros_like(x)
        identity = np.identity(len(x))
        for i in range(len(x)):
            gradient[i] = (f(x + h * identity[i]) - f(x - h * identity[i])) / (2 * h)
        return gradient

    def generate_random_sample(self, d):
        rng = np.random.default_rng()
        x = rng.random(d)
        return x * (self.rand_max - self.rand_min) + self.rand_min

    def fit(self, f: callable, d: int, maximize: bool = False) -> Tuple[np.ndarray, float]:
        """
        Finds the minimum or maximum of a function f using batch gradient descent starting from a random point.
        Parameters:
        ----------
        f : callable
            A callable function.
        d : int
            A positive integer representing the number of features.
        maximize : bool, optional
            If True, the method will find the maximum of the function. Otherwise, the default is False, and the method
            will find the minimum of the function.
        Returns:
        -------
        x : np.ndarray
            The parameter values at the minimum or maximum of the function.
        f(x) : float
            The value of the function at the minimum or maximum.
        """

        assert callable(f), "f must be a callable function"
        assert isinstance(d, int) and d > 0, "d should be a positive integer"

        x = self.generate_random_sample(d)
        sign = 1 if maximize else -1
        for i in range(self.max_iter):
            grad = self.gradient(f, x)
            if np.linalg.norm(grad) < self.tol:
                break
            x += sign * self.learning_rate * grad
        else:
            warnings.warn("Gradient failed to converge within the maximum number of iterations.")
        return x, f(x)

    def fit_multiple(self, f: callable, d: int, num_runs: int = 10, maximize: bool = False) -> Tuple[np.ndarray, float]:
        """
        Perform multiple runs of the optimization routine and return the best result.
        Parameters:
        -----------
        objective_func : callable
            The function to be optimized.
        dimension : int
            The dimensionality of the optimization problem.
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
            if self.fit(f, d, maximize)[1] < min_val:
                best_solution, min_val = self.fit(f, d, maximize)
        return best_solution, min_val


class MiniBatchGD(GradientDescent):
    """
    Mini-batch gradient descent optimizer.
    Attributes:
    ----------
    batch_size : int, optional
        The size of the mini-batch. Default is 10.
    learning_rate : float, optional
        The learning rate. Default is 0.1.
    max_iter : int, optional
        The maximum number of iterations. Default is 1000.
    tol : float, optional
        The tolerance for convergence. Default is 1e-8.
    Methods:
    -------
    gradient(f: callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        Computes the gradient of a function f at point x.
    fit(self, f: callable, d: int, maximize: bool = False) -> Tuple[np.ndarray, float]:
        Finds the minimum or maximum of a function f using gradient descent starting from a random point.
    """

    def __init__(self, batch_size=10, learning_rate=0.1, max_iter=100000, tol=1e-8,
                 rand_min: float = 0, rand_max: float = 1):
        super().__init__(learning_rate, max_iter, tol, rand_min, rand_max)
        self.batch_size = batch_size

    def fit(self, f: callable, d: int, maximize: bool = False) -> Tuple[np.ndarray, float]:
        """
        Finds the minimum or maximum of a function f using mini-batch gradient descent starting from a random point.
        Parameters:
        ----------
        f : callable
            A callable function.
        d : int
            A positive integer representing the number of features.
        maximize : bool, optional
            If True, the method will find the maximum of the function. Otherwise, the default is False, and the method
            will find the minimum of the function.
        Returns:
        -------
        x : np.ndarray
            The parameter values at the minimum or maximum of the function.
        f(x) : float
            The value of the function at the minimum or maximum.
        """
        assert callable(f), "f must be a callable function"
        assert isinstance(d, int) and d > 0, "d should be a positive integer"

        x = self.generate_random_sample(d)
        sign = 1 if maximize else -1
        f_old = f(x)
        for i in range(self.max_iter):
            indices = np.random.choice(d, self.batch_size)
            grad = self.gradient(f, x[indices])
            x[indices] += sign * self.learning_rate * grad
            f_new = f(x)
            if np.abs(f_new - f_old) < self.tol:
                break
            f_old = f_new
        else:
            warnings.warn("Gradient failed to converge within the maximum number of iterations.")

        return x, f(x)


class SGD(MiniBatchGD):
    """
    Stochastic Gradient Descent.
    Attributes:
    ----------
    learning_rate : float
        The learning rate to use in the gradient descent update.
    tol : float
        The convergence threshold for the norm of the gradient.
    max_iter : int
        The maximum number of iterations to run before stopping.
    Methods:
    -------
    gradient(f: callable, x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        Computes the gradient of a function f at point x.
    fit(self, f: callable, d: int, maximize: bool = False) -> Tuple[np.ndarray, float]:
        Finds the minimum or maximum of a function f using gradient descent starting from a random point.
    """

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 100000, tol: float = 1e-8,
                 rand_min: float = 0, rand_max: float = 1):
        super().__init__(batch_size=1, learning_rate=learning_rate, max_iter=max_iter, tol=tol,
                         rand_min=rand_min, rand_max=rand_max)


class ConjugateGradient(GradientDescent):
    """
    Initialize ConjugateGradient object.
    Parameters:
    -----------
    max_iter : int, optional
        The maximum number of iterations to run before stopping. Default is 100000.
    tol : float, optional
        The convergence threshold for the norm of the gradient. Default is 1e-8.
    rand_min : float, optional
        The minimum value for initializing random parameters. Default is 0.
    rand_max : float, optional
        The maximum value for initializing random parameters. Default is 1.
    """
    def __init__(self, max_iter: int = 100000, tol: float = 1e-8,
                 rand_min: float = 0, rand_max: float = 1):
        super().__init__(max_iter=max_iter, tol=tol, rand_min=rand_min, rand_max=rand_max)

    def fit(self, f: callable, d: int, maximize: bool = False) -> Tuple[np.ndarray, float]:
        """
        Find the minimum or maximum of a function using Conjugate Gradient method.
        Parameters:
        -----------
        f : callable
            The function to minimize or maximize.
        d : int
            The number of parameters in the function.
        maximize : bool, optional
            If True, finds the maximum of the function instead of the minimum. Default is False.
        Returns:
        --------
        x : np.ndarray
            The parameter values at the minimum or maximum of the function.
        f(x) : float
            The value of the function at the minimum or maximum.
        """
        x = self.generate_random_sample(d)
        sign = 1 if maximize else -1
        r = -self.gradient(f, x)
        p = r
        for i in range(self.max_iter):
            Ap = self.gradient(f, p)
            alpha = np.dot(r, r) / np.dot(p, Ap)
            x += alpha * p
            r_new = sign * self.gradient(f, x)
            if np.linalg.norm(r_new) < self.tol:
                break
            beta = np.dot(r_new, r_new) / np.dot(r, r)
            p = r_new + beta * p
            r = r_new

        return x, f(x)
