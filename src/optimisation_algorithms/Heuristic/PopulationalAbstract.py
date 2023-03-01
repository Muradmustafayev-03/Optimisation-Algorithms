from typing import Tuple
import numpy as np
from abc import ABC, abstractmethod


class PopulationalOptimization(ABC):
    def __init__(self, f: callable, d: int, population_size: int, tol: float = 1e-8, patience: int = 10**3,
                 max_iter: int = 10 ** 5, rand_min: float = 0, rand_max: float = 1):
        self.f = f
        self.d = d
        self.tol = tol
        self.patience = patience
        self.max_iter = max_iter
        self.rand_min = rand_min
        self.rand_max = rand_max
        self.population_size = population_size
        self.population = self.generate_population()

    def eval(self, population: np.array) -> np.ndarray:
        """
        Evaluates the population to the function.

        Parameters:
        ----------
        - population : numpy.ndarray
            An array representing the population to evaluate.

        Returns:
        -------
        - numpy.ndarray:
             An array representing the results of evaluating each solution in the harmony memory.
        """
        return np.apply_along_axis(self.f, 1, population)

    def generate_population(self):
        """
        Generates an initial population
        """
        return np.random.uniform(self.rand_min, self.rand_max, size=(self.population_size, self.d))

    def _check_improved(self, fitness, improvement_counter, best_fitness, best_solution, maximize):
        is_better = fitness > best_fitness if maximize else fitness < best_fitness
        if np.any(is_better):
            improvement_counter = 0
            best_fitness = fitness[is_better][0]
            best_solution = self.population[is_better][0]
        else:
            improvement_counter += 1
        return improvement_counter, best_fitness, best_solution

    @abstractmethod
    def update_population(self, **kwargs):
        """

        :param kwargs:
        :return:
        """

    def fit(self, maximize: bool = False) -> Tuple[np.ndarray, float]:
        """
        Finds the optimal solution for the given objective function.

        Parameters:
        ----------
        - maximize : bool (default: False)
            If True, the method will find the maximum of the function. Otherwise, the default is False, and the method
            will find the minimum of the function.
        Returns:
        -------
        - best_hm : numpy.ndarray
            An array representing the decision variables that optimize the objective function.
        - best_val : float
            The optimized function value.
        """
        best_fitness = -np.inf if maximize else np.inf
        best_solution = None
        improvement_counter = 0

        for _ in range(self.max_iter):
            fitness = self.eval(self.population)
            improvement_counter, best_fitness, best_solution = self._check_improved(
                fitness, improvement_counter, best_fitness, best_solution, maximize)
            if improvement_counter >= self.patience:
                break

            self.update_population(fitness=fitness, maximize=maximize)

        return best_solution, best_fitness
