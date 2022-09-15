import numpy as np


def gradient_descent(gradient, d, alpha=0.0002, tc_value=10 ** (-20), max_iterations=10000000, _range=2):
    current = (np.random.rand(d) - 0.5) * _range

    for _ in range(max_iterations):
        step = alpha * gradient(current)

        if step.all() < tc_value:
            break

        current = current - step

    return current
