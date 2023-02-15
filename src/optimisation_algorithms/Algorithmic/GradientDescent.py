import numpy as np


class GradientDescent:
    """
    Batch Gradient Descent.
    """

    def __init__(self, learning_rate: float = 0.1, max_iter: int = 1000, tol: float = 1e-8):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    @staticmethod
    def gradient(f, x, h=1e-8):
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

    def fit(self, f, d):
        """
        Finds the minimum of a function f using gradient descent starting from x0.
        """
        assert callable(f), "f must be a callable function"
        assert isinstance(d, int) and d > 0, "d should be a positive integer"

        x = np.random.rand(d)
        for i in range(self.max_iter):
            grad = self.gradient(f, x)
            if np.linalg.norm(grad) < self.tol:
                break
            x -= self.learning_rate * grad
        return x, f(x)


class MiniBatchGD(GradientDescent):
    """
    Mini-batch Gradient Descent.
    """

    def __init__(self, batch_size=10, learning_rate=0.1, max_iter=1000, tol=1e-8):
        super().__init__(learning_rate, max_iter, tol)
        self.batch_size = batch_size

    def fit(self, f, d):
        """
        Finds the minimum of a function f using mini-batch gradient descent starting from x0.
        """
        assert callable(f), "f must be a callable function"
        assert isinstance(d, int) and d > 0, "d should be a positive integer"

        x = np.random.rand(d)
        f_old = f(x)
        for i in range(self.max_iter):
            indices = np.random.choice(d, self.batch_size)
            grad = self.gradient(f, x[indices])
            x[indices] -= self.learning_rate * grad
            f_new = f(x)
            if np.abs(f_new - f_old) < self.tol:
                break
            f_old = f_new

        return x, f(x)


class SGD(MiniBatchGD):
    """
    Stochastic Gradient Descent.
    """

    def __init__(self, learning_rate=0.1, max_iter=1000, tol=1e-8):
        super().__init__(batch_size=1, learning_rate=learning_rate, max_iter=max_iter, tol=tol)
