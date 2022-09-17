from random import random
import numpy as np
from copy import deepcopy


def harmony_search(func: callable, d=30, hms=30, tc_value=10 ** (-16), max_iterations=10000, _range=1000,
                   hmcr=0.8, par=0.4, bw=2):

    hm = (np.random.rand(hms, d) - 0.5) * _range
    results = np.apply_along_axis(func, 1, hm)

    for _ in range(max_iterations):
        results = np.apply_along_axis(func, 1, hm)
        worst_i = np.argmax(results)
        best_i = np.argmin(results)

        if float(results[best_i]) < tc_value:
            break

        new_harmony = deepcopy(hm)

        for row in range(hms):
            new_harmony[row][random() > hmcr] = (random() - 0.5) * _range
            new_harmony[row][random() > par] += bw * (2 * random() - 1)

        new_results = np.apply_along_axis(func, 1, new_harmony)
        new_best_i = np.argmin(new_results)

        if float(new_results[new_best_i]) < float(results[worst_i]):
            hm[worst_i] = new_harmony[new_best_i]
            results = np.apply_along_axis(func, 1, hm)

    best_i = np.argmin(results)
    return hm[best_i]
