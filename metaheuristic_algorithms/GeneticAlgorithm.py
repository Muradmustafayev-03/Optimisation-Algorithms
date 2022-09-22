import numpy as np


class GeneticAlgorithm:
    def __init__(self, func: callable, d: int):
        self.func = func
        self.d = d

    def generate_population(self, size: int, _range=1):
        return (np.random.rand(size, self.d) - 0.5) * _range

    def eval(self, population: np.array):
        return np.apply_along_axis(self.func, 1, population)

    def select_elite(self, population: np.array,  elite_number: int):
        fitness = self.eval(population)
        indices = np.argpartition(fitness, -elite_number)[-elite_number:]
        return population[indices]

    def make_pairs(self, elite: np.array):
        return [[x1, x2] for x1 in elite for x2 in elite if self.func(x1) < self.func(x2)]

    @staticmethod
    def crossover(pairs: np.array, points: set):
        pivots = [int(p * len(pairs[0][0])) for p in points]
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
                odd = even+1

                individual1 += list(pair[0][pivots[even]:pivots[odd]])
                individual2 += list(pair[1][pivots[even]:pivots[odd]])
                try:
                    individual1 += list(pair[1][pivots[odd]:pivots[odd+1]])
                    individual2 += list(pair[0][pivots[odd]:pivots[odd + 1]])
                except IndexError:
                    individual1 += list(pair[1][pivots[odd]:])
                    individual2 += list(pair[0][pivots[odd]:])

            population.append(individual1)
            population.append(individual2)

        return np.array(population)

    def mutate(self):
        pass

    def evolve(self):
        pass


# pp = np.array([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3], [4, 4, 4, 4], [5, 5, 5, 5], [6, 6, 6, 6]])
# f = np.array([1, 2, 3, 4, 5, 6])
#
# e = GeneticAlgorithm.select_elite(pp, f, 4)
# ps = GeneticAlgorithm.make_pairs(e, lambda x: x[0])
# print(GeneticAlgorithm.crossover(ps, {0.25, 0.5, 0.75}))
