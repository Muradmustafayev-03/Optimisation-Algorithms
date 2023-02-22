from typing import Tuple
import numpy as np
from PopulationalAbstract import PopulationalOptimization


class AntColonyOptimization(PopulationalOptimization):
    """
    Ant Colony Optimization optimization algorithm.

    Parameters:
    ----------
    - f : callable
        The objective function to be optimized.
    - d : int
        The dimensionality of the decision variables.
    - pop_size : int (default=50)
        The size of the ant colony.
    - alpha : float (default=1)
        The relative importance of the pheromone trail in path selection.
    - beta : float (default=2)
        The relative importance of the distance in path selection.
    - evaporation_rate : float (default=0.5)
        The rate at which pheromone evaporates over time.
    - tolerance : float (default=1e-8)
        The convergence threshold.
    - patience : int (default=10)
        The number of iterations to wait for improvement before stopping the optimization.
    - max_iter : int (default=10 ** 5)
        The maximum number of iterations to run.
    - rand_min : float (default=0)
        The minimum value for random initialization of decision variables.
    - rand_max : float (default=1)
        The maximum value for random initialization of decision variables.

    Methods:
    --------
    - eval(pop: np.ndarray) -> np.ndarray:
        Evaluates the objective function at each point in the population.
    - generate_population()
        Generates an initial ant colony.
    - select_path(current: int, unvisited: List[int], pheromone: np.ndarray, distances: np.ndarray,
                  alpha: float, beta: float) -> int:
        Selects the next node for an ant to visit using a probabilistic model.
    - get_solution(pheromone: np.ndarray, distances: np.ndarray) -> Tuple[np.ndarray, float]:
        Returns the best solution found by the ant colony.
    - fit(maximize: bool = False) -> Tuple[np.ndarray, float]:
        Finds the optimal solution for the given objective function.
    """
    def __init__(self, f: callable, d: int, pop_size: int = 50, alpha: float = 1, beta: float = 2,
                 evaporation_rate: float = 0.5, tolerance: float = 1e-8, patience: int = 10,
                 max_iter: int = 10 ** 5, rand_min: float = 0, rand_max: float = 1):
        self.f = f
        self.d = d
        self.population_size = pop_size
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.tolerance = tolerance
        self.patience = patience
        self.max_iter = max_iter
        self.rand_min = rand_min
        self.rand_max = rand_max
