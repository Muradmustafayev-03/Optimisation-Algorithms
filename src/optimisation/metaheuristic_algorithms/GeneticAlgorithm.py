import random
import numpy as np


class GeneticAlgorithm:
    """
        Optimization using metaheuristic Genetic Algorithm,
        that imitates natural evolution to find the fittest individual

        Attributes
        ----------
        func : callable
            Function to minimize
        d: int
            Number of dimensions of the function [f(x) = f(x_1, ..., x_d)]
        _range: float
            Range of values for the initial population x_1 in (-range/2; range/2)

        Methods
        -------
        generate_population(size: int, _range: float = 1)
            Generates an initial population.

        eval(population: numpy.array)
            Evaluates the given population to the function.

        select_best(population: numpy.array, n: int)
            Selects n best (fittest) individuals from the population.

        make_pairs(elite: numpy.array)
            Makes all possible pairs out of the given population.

        crossover(pairs: numpy.array, points: set = (0, 0.5, 1))
            Does a crossover at the given points between parents to get 2 children from each pair.

        breed(population, n_breed: int, n_remain: int, crossover_points: set = (0, 0.5, 1))
            Generates a new population by replacing the worst individuals of an old generation
            with the best individuals of the new generation.

        mutate(population, individuals_rate=0.5, genes_rate=0.1, _range: float = 1.)
            Applies some random changes to the population. Can be applied either to the whole
            population, or to the newly created generation.

        evolve(self, population_size: int = 1000, n_breed: int = 200, n_remain: int = 200,
        crossover_points: set = (0, 0.5, 1), individuals_mutation_rate: float = 0.5,
        genes_mutation_rate: float = 0.1, mutation_range: float = 1.,
        max_iterations: int = 10000, _terminate: int = 1000):
            Finds the global minima of the function by simulating an evolving population.
        :return:
        """

    def __init__(self, func: callable, d: int, _range: float = 1000):
        """
        Parameters
        ----------
        func : callable
            Function to minimize
        d: int
            Number of dimensions of the function [f(x) = f(x_1, ..., x_d)]
        _range: float
            Range of values for the initial population x_1 in (-range/2; range/2)

        :return: None
        """
        self.func = func
        self.d = d
        self.range = _range

    def generate_population(self, size: int, _range: float = 1):
        """
        Generates an initial population

        Parameters
        ----------
        size: int
            Number of individuals in the population
        _range: float
            Range of values for individuals of the population x_1 in (-range/2; range/2)

        Returns
        -------
        :return: numpy.array: population
        """
        return (np.random.rand(size, self.d) - 0.5) * _range

    def eval(self, population: np.array):
        """
        Evaluates the given population to the function

        Parameters
        ----------
        population: numpy.array
            Population to evaluate

        Returns
        -------
        :return: numpy.array: fittness values of all the individuals
        """
        return np.apply_along_axis(self.func, 1, population)

    def select_best(self, population: np.array, n: int):
        """
        Selects n best (fittest) individuals from the population.

        Parameters
        ----------
        population: numpy.array \n
        n: int

        Returns
        -------
        :return: numpy.array: n the fittest individuals from the population
        """
        fitness = list(self.eval(population))
        indices = sorted(range(len(fitness)), key=lambda sub: fitness[sub])[:n]
        return population[indices]

    @staticmethod
    def make_pairs(elite: np.array):
        """
        Makes all possible pairs out of the given population.

        Parameters
        ----------
        elite: numpy.array
            Population or an array of the best individuals of the population

        Returns
        -------
        :return: numpy.array
            Array of pairs of the shape (n_p, 2, d), n_p = n*(n-1)/2 where n = len(elite)
        """
        return [[elite[i], elite[j]] for i in range(len(elite)) for j in range(len(elite)) if i < j]

    def __pivots(self, points: set):
        pivots = [int(point * self.d) for point in points]
        if 0 not in pivots:
            pivots.append(0)
        if self.d not in pivots:
            pivots.append(self.d)
        pivots.sort()

        return pivots

    @staticmethod
    def __crossover(pairs: np.array, pivots: list):
        population = []

        for pair in pairs:
            child_1 = []
            child_2 = []

            for pivot in pivots[:-1]:
                index = pivots.index(pivot)
                next_pivot = pivots[index + 1]

                if index % 2 == 0:
                    child_1 += list(pair[0][pivot:next_pivot])
                    child_2 += list(pair[1][pivot:next_pivot])
                else:
                    child_1 += list(pair[1][pivot:next_pivot])
                    child_2 += list(pair[0][pivot:next_pivot])

            population.append(np.array(child_1))
            population.append(np.array(child_2))

        return np.array(population)

    def crossover(self, pairs: np.array, points: set = (0, 0.5, 1)):
        """
        Does a crossover at the given points between parents to get 2 children from each pair.

        Parameters
        ----------
        pairs: numpy.array
            Array of pairs of the shape (m, 2, d)
        points: set
            The set of relative points in range(0, 1), by default splits by half

        Returns
        -------
        :return: numpy.array
            New generation
        """

        pivots = self.__pivots(points)
        return self.__crossover(pairs, pivots)

    def uniform_crossover(self, pairs: np.array):
        """
        Does a crossover at the given points between parents to get 2 children from each pair.

        Parameters
        ----------
        pairs: numpy.array
            Array of pairs of the shape (m, 2, d)

        Returns
        -------
        :return: numpy.array
            New generation
        """

        pivots = list(range(self.d))
        return self.__crossover(pairs, pivots)

    def breed(self, population: np.array, n_breed: int, n_remain: int, crossover_points: set = (0, 0.5, 1)):
        """
        Generates a new population by replacing the worst individuals of an old generation
        with the best individuals of the new generation.

        Parameters
        ----------
        population: numpy.array
            Initial population
        n_breed: int
            Number of elite to select and breed
        n_remain: int
            Number of elite to remain, not to replace with the new generation
        crossover_points: set
            The set of relative crossover points in range(0, 1), by default splits by half

        Returns
        -------
        :return: numpy.array
            New generation
        """
        elite = self.select_best(population, n_breed)
        pairs = self.make_pairs(elite)
        new_generation = self.crossover(pairs, crossover_points)

        new_population = list(self.select_best(population, n_remain)) + \
                         list(self.select_best(new_generation, len(population) - n_remain))

        return np.array(new_population)

    def mutate(self, population: np.array, individuals_rate: float = 0.5, genes_rate: float = 0.1, _range: float = 1.):
        """
        Applies some random changes to the population. Can be applied either to the whole
        population, or to the newly created generation.

        Parameters
        ----------
        population: numpy.array
            Initial population to mutate
        individuals_rate: float
            Share of individuals to mutate
        genes_rate: float
            Share of genes to change
        _range: float
            Mutation range

        Returns
        -------
        :return: numpy.array: mutated population
        """
        for individual in range(len(population)):
            if individuals_rate > random.random():
                for gene in range(self.d):
                    if genes_rate > random.random():
                        population[individual][gene] += (random.random() - 0.5) * _range
        return population

    def evolve(self, population_size: int = 1000, n_breed: int = 200, n_remain: int = 200,
               crossover_points: set = (0, 0.5, 1), individuals_mutation_rate: float = 0.5,
               genes_mutation_rate: float = 0.1, mutation_range: float = 1.,
               max_iterations: int = 10000, _terminate: int = 1000):
        """
        Finds the global minima of the function by simulating an evolving population.

        Parameters
        ----------
        population_size: int
            Number of individuals in the population
        n_breed: int
            Number of elite to select and breed
        n_remain: int
            Number of elite to remain, not to replace with the new generation
        crossover_points: set
            The set of relative crossover points in range(0, 1), by default splits by half
        individuals_mutation_rate: float
            Share of individuals to mutate
        genes_mutation_rate: float
            Share of genes to change
        mutation_range: float
            Mutation range
        max_iterations: int
            Maximal number of iterations
        _terminate: int
            Maximal number of iterations when the best individual doesn't change

        Returns
        -------
        :return: numpy.array
            x_min = (x_1, ..., x_d) where f(x_min) = min f(x)
        """

        population = self.generate_population(population_size, self.range)

        best_i = np.argmin(self.eval(population))
        to_terminate = _terminate
        for _ in range(max_iterations):
            population = self.breed(population, n_breed, n_remain, crossover_points)
            population = self.mutate(population, individuals_mutation_rate, genes_mutation_rate, mutation_range)

            if np.argmin(population) < np.argmin(self.eval(population)):
                best_i = np.argmin(self.eval(population))
                to_terminate = _terminate
            else:
                to_terminate -= 1

            if to_terminate == 0:
                break

        return population[best_i]
