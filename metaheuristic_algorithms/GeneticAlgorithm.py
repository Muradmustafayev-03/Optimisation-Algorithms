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
        indices = np.argpartition(fitness, -n)[-n:]
        return population[indices]

    def select_worst(self, population: np.array, n: int):
        fitness = self.eval(population)
        indices = np.argpartition(fitness, n)[:n]
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

    def replace_worst(self, population, n: int, crossover_points: set):
        elite = self.select_best(population, n)
        pairs = self.make_pairs(elite)
        new_generation = self.crossover(pairs, crossover_points)
        new_elite = self.select_best(new_generation, n)
        worst = self.select_worst(population, n)

        for i in range(n):
            population[population == worst[i]] = new_elite[i]

        return population

    def mutate(self, population):
        np.random.shuffle(population)

    def evolve(self):
        pass


c = GeneticAlgorithm(lambda x: sum(x), 10)
p = c.generate_population(12)
print(c.eval(p))
newp = c.replace_worst(p, 4, {0.25, 0.5, 0.75})
print(c.eval(newp))

