from GradientDescentAbstract import BatchGD as Batch, StochasticGD as Stochastic, MiniBatchGD as MiniBatch
from GradientDescentAbstract import SimpleGD as Simple, ConjugateGD as Conjugate


class GradientDescent(Simple, Batch):
    def __init__(self, f: callable, d: int, learning_rate: float = 0.1, max_iter: int = 10 ** 5, tol: float = 1e-8,
                 h: float = 1e-8, rand_min: float = 0, rand_max: float = 1):
        Simple.__init__(self, f, d, learning_rate, max_iter, tol, h, rand_min, rand_max)


class StochasticGradientDescent(Simple, Stochastic):
    def __init__(self, f: callable, d: int, learning_rate: float = 0.1, max_iter: int = 10 ** 5, tol: float = 1e-8,
                 h: float = 1e-8, rand_min: float = 0, rand_max: float = 1):
        Simple.__init__(self, f, d, learning_rate, max_iter, tol, h, rand_min, rand_max)
        Stochastic.__init__(self, d, tol)


class MiniBatchGradientDescent(Simple, MiniBatch):
    def __init__(self, f: callable, d: int, batch_size: int, learning_rate: float = 0.1, max_iter: int = 10 ** 5,
                 tol: float = 1e-8, h: float = 1e-8, rand_min: float = 0, rand_max: float = 1):
        Simple.__init__(self, f, d, learning_rate, max_iter, tol, h, rand_min, rand_max)
        MiniBatch.__init__(self, d, batch_size, tol)


class ConjugateGradientDescent(Conjugate, Batch):
    def __init__(self, f: callable, d: int, max_iter: int = 10 ** 5, tol: float = 1e-8, h: float = 1e-8,
                 rand_min: float = 0, rand_max: float = 1):
        Conjugate.__init__(self, f, d, max_iter, tol, h, rand_min, rand_max)


class ConjugateSGD(Conjugate, Stochastic):
    def __init__(self, f: callable, d: int, max_iter: int = 10 ** 5, tol: float = 1e-8, h: float = 1e-8,
                 rand_min: float = 0, rand_max: float = 1):
        Conjugate.__init__(self, f, d, max_iter, tol, h, rand_min, rand_max)
        Stochastic.__init__(self, d, tol)


class MiniBatchConjugateGD(Conjugate, MiniBatch):
    def __init__(self, f: callable, d: int, batch_size: int, max_iter: int = 10 ** 5, tol: float = 1e-8,
                 h: float = 1e-8, rand_min: float = 0, rand_max: float = 1):
        Conjugate.__init__(self, f, d, max_iter, tol, h, rand_min, rand_max)
        MiniBatch.__init__(self, d, batch_size, tol)
