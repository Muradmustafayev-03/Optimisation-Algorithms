from random import random
import numpy as np
from copy import deepcopy


class HarmonySearch:
    def __init__(self, func: callable, d: int, _range: float = 1000):
        self.func = func
        self.d = d
        self.range = _range

    def generate_harmony(self, hms: int):
        return (np.random.rand(hms, self.d) - 0.5) * self.range

    def eval(self, hm: np.array):
        return np.apply_along_axis(self.func, 1, hm)
    
    def improvise(self, hm, hmcr, par, bw):
        new_hm = deepcopy(hm)

        for row in range(len(hm)):
            new_hm[row][random() > hmcr] = (random() - 0.5) * self.range
            new_hm[row][random() > par] += bw * (2 * random() - 1)

        return new_hm

    def change_harmony(self, old_hm, new_hm):
        old_results = self.eval(old_hm)
        new_results = self.eval(new_hm)

        old_worst_i = np.argmax(old_results)
        new_best_i = np.argmin(new_results)

        if float(new_results[new_best_i]) < float(old_results[old_worst_i]):
            old_hm[old_worst_i] = new_hm[new_best_i]

        return old_hm

    def find_harmony(self, hms=30, max_iterations=10000, hmcr=0.8, par=0.4, bw=2):
        hm = self.generate_harmony(hms)
        results = self.eval(hm)

        for _ in range(max_iterations):
            new_hm = self.improvise(hm, hmcr, par, bw)
            hm = self.change_harmony(hm, new_hm)
            results = self.eval(hm)

        best_i = np.argmin(results)
        return hm[best_i]
