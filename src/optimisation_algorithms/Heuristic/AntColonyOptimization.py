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

    @staticmethod
    def select_path(pheromone_matrix: np.ndarray, distance_matrix: np.ndarray, alpha: float, beta: float) -> np.ndarray:
        """
        Selects a path for each ant based on pheromone trail strength and distance.

        Parameters:
        ----------
        - pheromone_matrix : np.ndarray of shape (n, n)
            The pheromone matrix.
        - distance_matrix : np.ndarray of shape (n, n)
            The distance matrix.
        - alpha : float
            The pheromone influence parameter.
        - beta : float
            The distance influence parameter.

        Returns:
        -------
        - paths : np.ndarray of shape (n_ants, n)
            The paths chosen by each ant.
        """
        n_ants = pheromone_matrix.shape[0]
        n = pheromone_matrix.shape[1]
        paths = np.zeros((n_ants, n), dtype=int)
        visited = np.zeros(n, dtype=bool)
        for ant in range(n_ants):
            current_node = np.random.randint(n)
            visited[current_node] = True
            paths[ant, 0] = current_node
            for i in range(1, n):
                probs = ((pheromone_matrix[current_node] ** alpha) *
                         ((1.0 / distance_matrix[current_node]) ** beta))
                probs *= 1.0 - visited
                probs /= np.sum(probs)
                current_node = np.random.choice(np.arange(n), p=probs)
                visited[current_node] = True
                paths[ant, i] = current_node
            visited[:] = False
        return paths

    def fit(self, maximize=False):
        population = self.generate_population()

        best_fitness = -np.inf if maximize else np.inf
        best_solution = None
        improvement_counter = 0

        for iteration in range(self.max_iter):
            fitness = self.eval(population)
            improvement_counter, best_fitness, best_solution = \
                self.check_improved(population, fitness, improvement_counter, best_fitness, best_solution, maximize)
            if improvement_counter >= self.patience:
                break

            pheromone = np.zeros((self.d, self.d))
            distances = np.apply_along_axis(lambda x: np.linalg.norm(x - population, axis=1), 1, population)
            for ant in range(self.population_size):
                paths = self.select_path(pheromone, distances, self.alpha, self.beta)
                path_distances = np.apply_along_axis(lambda x: np.sum(distances[range(self.d), x[:-1], x[1:]]), 1,
                                                     paths)
                path_fitness = self.eval(population[ant][paths])
                is_better = path_fitness > best_fitness if maximize else path_fitness < best_fitness
                if np.any(is_better):
                    best_fitness = path_fitness[is_better][0]
                    best_solution = population[ant][paths[is_better][0]]
                for i in range(self.d):
                    for j in range(self.d):
                        pheromone[i, j] *= self.evaporation_rate
                        pheromone[i, j] += np.sum(1.0 / path_distances * path_fitness * (paths[:, i] == j))
        return best_solution, best_fitness
