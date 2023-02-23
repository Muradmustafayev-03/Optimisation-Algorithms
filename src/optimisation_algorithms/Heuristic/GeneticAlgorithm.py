from PopulationalAbstract import PopulationalOptimization
from typing import Tuple
import numpy as np


class GeneticAlgorithm(PopulationalOptimization):
    """
    Genetic Algorithm optimization algorithm.

    Parameters:
    ----------
    - f : callable
        The objective function to be optimized.
    - d : int
        The dimensionality of the decision variables.
    - pop_size : int (default=50)
        The size of the population.
    - mutation_rate : float (default=0.1)
        The mutation rate.
    - crossover_rate : float (default=0.8)
        The crossover rate.
    - n_elites : int (default=1)
        The number of elite solutions to keep in each generation.
    - tol : float (default=1e-8)
        The convergence threshold.
    - patience : int (default=10**3)
        The number of iterations to wait for improvement before stopping the optimization.
    - max_iter : int (default=10**5)
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
        Generates an initial population.
    - select(pop: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        Selects individuals for mating based on their fitness.
    - crossover(parents: np.ndarray) -> np.ndarray:
        Combines the decision variables of two parents to create a new offspring.
    - mutate(individual: np.ndarray) -> np.ndarray:
        Mutates an individual by randomly adjusting its decision variables.
    - elitism(pop: np.ndarray, fitness: np.ndarray, n_elites: int) -> np.ndarray:
        Selects the elite solutions from the population.
    - fit(maximize: bool = False) -> Tuple[np.ndarray, float]:
        Finds the optimal solution for the given objective function.
    """
    def __init__(self, f: callable, d: int, pop_size: int = 50, mutation_rate: float = 0.1, crossover_rate: float = 0.8,
                 n_elites: int = 1, tol: float = 1e-8, patience: int = 10**3, max_iter: int = 10**5,
                 rand_min: float = 0, rand_max: float = 1):
        self.f = f
        self.d = d
        self.population_size = pop_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.n_elites = n_elites
        self.tol = tol
        self.patience = patience
        self.max_iter = max_iter
        self.rand_min = rand_min
        self.rand_max = rand_max

    def select(self, fitness: np.ndarray) -> np.ndarray:
        """
        Selects individuals for mating based on their fitness.

        Parameters:
        ----------
        - pop : numpy.ndarray
            A numpy array representing the population.
        - fitness : numpy.ndarray
            A numpy array representing the fitness values of each individual in the population.

        Returns:
        -------
        - numpy.ndarray:
            A numpy array representing the selected individuals for mating.
        """
        idx = np.random.choice(self.population_size, self.population_size, p=fitness/fitness.sum())
        return idx

    def crossover(self, parents: np.ndarray) -> np.ndarray:
        """
        Combine the decision variables of two parents to create a new offspring.

        Parameters:
        ----------
        - parents : np.ndarray of shape (2, self.d)
            The decision variables of the two parents.

        Returns:
        -------
        - child : np.ndarray of shape (self.d,)
            The decision variables of the new offspring.
        """
        crossover_point = np.random.randint(self.d)
        offspring = np.concatenate([parents[0][:crossover_point], parents[1][crossover_point:]])
        return offspring

    def mutate(self, individual: np.ndarray) -> np.ndarray:
        """
        Mutates an individual by randomly adjusting its decision variables.

        Parameters:
        ----------
        - individual : np.ndarray
            The individual to mutate.

        Returns:
        -------
        - np.ndarray
            The mutated individual.
        """
        mask = np.random.rand(*individual.shape) < self.mutation_rate
        mutation = np.random.uniform(self.rand_min, self.rand_max, size=individual.shape)
        individual[mask] = mutation[mask]
        return individual

    @staticmethod
    def elitism(pop: np.ndarray, fitness: np.ndarray, n_elites: int) -> np.ndarray:
        """
        Selects the elite solutions from the population.

        Parameters:
        ----------
        - pop : np.ndarray
            The population of solutions.
        - fitness : np.ndarray
            The fitness values of each solution in the population.
        - n_elites : int
            The number of elite solutions to select.

        Returns:
        -------
        - elite_pop : np.ndarray
            The elite solutions.
        """
        sorted_indices = np.argsort(fitness)
        elite_indices = sorted_indices[-n_elites:]
        elite_pop = pop[elite_indices, :]
        return elite_pop

    def fit(self, maximize=False) -> Tuple[np.ndarray, float]:
        # Generate initial population
        population = self.generate_population()

        # Initialize variables
        n_parents = int(np.ceil((self.population_size - self.n_elites) / 2))
        n_mutations = int(np.ceil(self.mutation_rate * self.population_size))
        n_crossovers = int(np.ceil(self.crossover_rate * self.population_size))
        best_fitness = -np.inf if maximize else np.inf
        best_solution = None
        improvement_counter = 0

        for _ in range(self.max_iter):
            fitness = self.eval(population)
            improvement_counter, best_fitness, best_solution = \
                self.check_improved(population, fitness, improvement_counter, best_fitness, best_solution, maximize)
            if improvement_counter >= self.patience:
                break

            # Select parents for mating
            parents_idx = self.select(fitness)[:n_parents]

            # Apply crossover and mutation to create new offspring
            crossovers_idx = np.random.choice(parents_idx, size=n_crossovers, replace=True)
            offspring = np.empty((self.population_size, self.d))
            offspring[:n_crossovers] = self.crossover(population[crossovers_idx])
            offspring[n_crossovers:self.n_elites] = population[parents_idx[:self.n_elites]]
            mutations_idx = np.random.choice(range(self.population_size), size=n_mutations, replace=False)
            offspring[mutations_idx] = self.mutate(offspring[mutations_idx])

            # Select elite solutions to keep
            elites = self.elitism(population, fitness, self.n_elites)
            offspring[:self.n_elites] = elites

            # Replace old population with new offspring
            population = offspring

        return best_solution, best_fitness
