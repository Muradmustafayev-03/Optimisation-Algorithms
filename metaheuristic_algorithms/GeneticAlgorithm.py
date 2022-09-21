import numpy as np


class GeneticAlgorithm:
    @staticmethod
    def generate_population(d: int, size: int, _range=1):
        return (np.random.rand(size, d) - 0.5) * _range

    @staticmethod
    def eval(func: callable, population: np.array):
        return np.apply_along_axis(func, 1, population)

    @staticmethod
    def select_elite(population: np.array, fitness: np.array, elite_number: int):
        indices = np.argpartition(fitness, -elite_number)[-elite_number:]
        return population[indices]

    @staticmethod
    def make_pairs(elite: np.array):
        return [[x1, x2] for x1 in elite for x2 in elite if x1 < x2]

    def crossover(self):
        pass

    def mutate(self):
        pass

    def evolve(self):
        pass


p = np.array([1, 2, 3, 4, 5, 6])
f = np.array([1, 2, 3, 4, 5, 6])

e = GeneticAlgorithm.select_elite(p, f, 4)
print(GeneticAlgorithm.make_pairs(e))
