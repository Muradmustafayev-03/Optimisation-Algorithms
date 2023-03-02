import numpy as np
from PopulationalAbstract import PopulationalOptimization


class AntColonyOptimization(PopulationalOptimization):
    def __init__(self, f: callable, d: int, min_val: float, max_val: float, step: float = 1, population_size: int = 20,
                 tol: float = 1e-8, patience: int = 10**3, max_iter: int = 10 ** 5, alpha: float = 1,
                 beta: float = 3, evaporation_rate: float = 0.5, pheromone_init: float = 0.01):
        super().__init__(f, d, population_size, tol, patience, max_iter, min_val, max_val)
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        self.step = step
        self.steps_range = int((max_val - min_val) / step)
        self.pheromones = np.full((self.steps_range, self.steps_range), pheromone_init)

    def eval_path(self, path):
        return sum(self.f([path[j], path[j + 1]]) for j in range(len(path) - 1))

    def update_population(self, **kwargs):
        pheromones = np.copy(self.pheromones)
        pheromones *= (1 - self.evaporation_rate)

        for i in range(self.population_size):
            path = self.generate_path(pheromones)
            path_fitness = self.eval_path(path)
            pheromones = self.update_pheromones(pheromones, path, path_fitness)

        self.pheromones = pheromones

    def generate_path(self, pheromones):
        allowed_moves = np.arange(self.rand_min, self.rand_max, self.step)
        start_node = self.make_move(None, allowed_moves)
        allowed_moves = np.setdiff1d(allowed_moves, start_node)
        path = [start_node]

        while allowed_moves.size > 1:
            current_node = path[-1]
            allowed_moves = np.setdiff1d(allowed_moves, current_node)
            move_probs = self.calculate_move_probs(current_node, allowed_moves, pheromones)
            next_node = self.make_move(move_probs, allowed_moves)
            path.append(next_node)

        return np.array(path)

    # def get_allowed_moves(self, current_node):
    #     allowed_moves = np.arange(self.rand_min, self.rand_max, self.step)
    #     allowed_moves = np.setdiff1d(allowed_moves, current_node)
    #     return allowed_moves

    def calculate_move_probs(self, current_node, allowed_moves, pheromones):
        move_probs = []
        denominator = 0

        for move in allowed_moves:
            pheromone = pheromones[current_node, move]
            print(current_node, move)
            distance = self.f([current_node, move])
            denominator += (pheromone ** self.alpha) * ((1 / distance) ** self.beta)

        for move in allowed_moves:
            pheromone = pheromones[current_node, move]
            distance = self.f([current_node, move])
            prob = (pheromone ** self.alpha) * ((1 / distance) ** self.beta) / denominator
            move_probs.append(prob)

        return np.array(move_probs).flatten()

    @staticmethod
    def make_move(move_probs, allowed_moves):
        return np.random.choice(allowed_moves, p=move_probs)

    @staticmethod
    def update_pheromones(pheromones, ant_path, ant_fitness):
        for i in range(len(ant_path) - 1):
            curr_node = ant_path[i]
            next_node = ant_path[i + 1]
            pheromones[curr_node, next_node] += ant_fitness
            pheromones[next_node, curr_node] = pheromones[curr_node, next_node]

        return pheromones


def dis(x):
    return abs(x[1] - x[0])


aco = AntColonyOptimization(dis, 2, -20, 20)
ant_path = aco.generate_path(aco.pheromones)
print("path: ", ant_path)
ant_fitness = sum(dis([ant_path[j], ant_path[j + 1]]) for j in range(len(ant_path) - 1))
print("fitness = ", ant_fitness)
