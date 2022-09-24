import random
import numpy as np


class GeneticAlgorithm:
    def __init__(self, func: callable, d: int):
        self.func = func
        self.d = d

    def generate_population(self, size: int, _range: float = 1):
        return (np.random.rand(size, self.d) - 0.5) * _range

    def eval(self, population: np.array):
        return np.apply_along_axis(self.func, 1, population)

    def select_best(self, population: np.array, n: int):
        fitness = list(self.eval(population))
        indices = sorted(range(len(fitness)), key=lambda sub: fitness[sub])[:n]
        return population[indices]

    @staticmethod
    def make_pairs(elite: np.array):
        return [[elite[i], elite[j]] for i in range(len(elite)) for j in range(len(elite)) if i < j]

    def crossover(self, pairs: np.array, points: set = (0, 0.5, 1)):
        pivots = [int(point * self.d) for point in points]
        if 0 not in pivots:
            pivots.append(0)
        if self.d not in pivots:
            pivots.append(self.d)
        pivots.sort()

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

    def breed(self, population, n_breed: int, n_remain: int, crossover_points: set = (0, 0.5, 1)):
        elite = self.select_best(population, n_breed)
        pairs = self.make_pairs(elite)
        new_generation = self.crossover(pairs, crossover_points)

        new_population = list(self.select_best(population, n_remain)) + \
                         list(self.select_best(new_generation, len(population) - n_remain))

        return np.array(new_population)

    def mutate(self, population, individuals_rate=0.5, genes_rate=0.1, _range: float = 1.):
        for individual in range(len(population)):
            if individuals_rate > random.random():
                for gene in range(self.d):
                    if genes_rate > random.random():
                        population[individual][gene] += (random.random() - 0.5) * _range
        return population

    def evolve(self, population_size: int = 1000, population_range: float = 10, n_breed: int = 200, n_remain: int = 200,
               crossover_points: set = (0, 0.5, 1), individuals_mutation_rate: float = 0.5,
               genes_mutation_rate: float = 0.1, mutation_range: float = 1., max_iterations: int = 10000):

        population = self.generate_population(population_size, population_range)

        best_i = np.argmin(self.eval(population))
        to_terminate = 100
        for _ in range(max_iterations):
            population = self.breed(population, n_breed, n_remain, crossover_points)
            population = self.mutate(population, individuals_mutation_rate, genes_mutation_rate, mutation_range)

            if np.argmin(population) < np.argmin(self.eval(population)):
                best_i = np.argmin(self.eval(population))
                to_terminate = 1000
            else:
                to_terminate -= 1

            if to_terminate == 0:
                break

        return population[best_i]
