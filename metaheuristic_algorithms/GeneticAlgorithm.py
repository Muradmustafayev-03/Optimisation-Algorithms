import random
import numpy as np


class GeneticAlgorithm:
    def __init__(self, func: callable, d: int):
        self.func = func
        self.d = d

    def generate_population(self, size: int, _range=1):
        return (np.random.rand(size, self.d) - 0.5) * _range

    def eval(self, population: np.array):
        return np.apply_along_axis(self.func, 1, population)

    def select_best(self, population: np.array, n: int):
        fitness = self.eval(population)
        indices = np.argpartition(fitness, n)[:n]
        return population[indices]

    def select_worst(self, population: np.array, n: int):
        fitness = self.eval(population)
        indices = np.argpartition(fitness, -n)[-n:]
        return population[indices]

    def make_pairs(self, elite: np.array):
        return [[x1, x2] for x1 in elite for x2 in elite if self.func(x1) < self.func(x2)]

    @staticmethod
    def crossover(pairs: np.array, points: set):
        pivots = [int(point * len(pairs[0][0])) for point in points]
        if 0 not in pivots:
            pivots.append(0)
        if len(pairs[0][0]) not in pivots:
            pivots.append(len(pairs[0][0]))
        pivots.sort()

        population = []

        for pair in pairs:
            individual1 = []
            individual2 = []

            for even in range(len(pivots))[:-1:2]:
                odd = even + 1

                individual1 += list(pair[0][pivots[even]:pivots[odd]])
                individual2 += list(pair[1][pivots[even]:pivots[odd]])
                try:
                    individual1 += list(pair[1][pivots[odd]:pivots[odd + 1]])
                    individual2 += list(pair[0][pivots[odd]:pivots[odd + 1]])
                except IndexError:
                    individual1 += list(pair[1][pivots[odd]:])
                    individual2 += list(pair[0][pivots[odd]:])

            population.append(individual1)
            population.append(individual2)

        return np.array(population)

    def replace_worst(self, population, n_select: int, n_replace: int, crossover_points: set):
        elite = self.select_best(population, n_select)
        pairs = self.make_pairs(elite)
        new_generation = self.crossover(pairs, crossover_points)
        new_elite = self.select_best(new_generation, n_replace)
        worst = self.select_worst(population, n_replace)

        for i in range(n_replace):
            population[population == worst[i]] = new_elite[i]

        return population

    @staticmethod
    def mutate(population, individuals_rate=0.5, genes_rate=0.1, _range: float = 1.):
        for individual in population:
            if individuals_rate > random.random():
                for gene in individual:
                    if genes_rate > random.random():
                        individual[individual == gene] += _range * random.random()
        return population

    def evolve(self, population_size, population_range, n_elite: int, n_replace: int,
               crossover_points: set = (0, 0.5, 1), individuals_mutation_rate: float = 0.5,
               genes_mutation_rate: float = 0.1, mutation_range: float = 1., max_iterations: int = 100000):

        population = self.generate_population(population_size, population_range)

        best_i = np.argmin(self.eval(population))
        to_terminate = 1000
        for _ in range(max_iterations):
            print(self.eval(population)[best_i])
            population = self.replace_worst(population, n_elite, n_replace, crossover_points)
            population = self.mutate(population, individuals_mutation_rate, genes_mutation_rate, mutation_range)

            if np.argmin(population) < np.argmin(self.eval(population)):
                best_i = np.argmin(self.eval(population))
                to_terminate = 1000
            else:
                to_terminate -= 1

            if to_terminate == 0:
                break

        return population[best_i]
