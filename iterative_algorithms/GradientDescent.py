import numpy as np


def grad(func, x, tc_value=10 ** (-8)):
    def replace(array, init, place):
        new = np.copy(array)
        new[new == init] = place
        return new

    return np.array([(func(replace(x, x_i, x_i + tc_value)) - func(x)) / tc_value for x_i in x])


def batch_gradient_descent(gradient: callable, d, alpha=0.2, tc_value=10 ** (-20), max_iterations=1000000, _range=1000):
    try:
        current = (np.random.rand(d) - 0.5) * _range

        for _ in range(max_iterations):
            step = alpha * gradient(current)

            if step.all() < tc_value:
                break
            current = current - step

        return current

    except RuntimeWarning:
        return batch_gradient_descent(gradient, d, alpha, tc_value, max_iterations, _range)


def approximated_gradient_descent(func: callable, d, alpha=0.2, tc_value=10 ** (-20), max_iterations=1000000,
                                  _range=1000):
    current = (np.random.rand(d) - 0.5) * _range

    for _ in range(max_iterations):
        step = alpha * grad(func, current)

        if step.all() < tc_value:
            break
        current = current - step

    return current
