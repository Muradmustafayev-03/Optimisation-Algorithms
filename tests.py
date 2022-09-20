from iterative_algorithms.GradientDescent import batch_gradient_descent, approximated_gradient_descent, grad
from metaheuristic_algorithms.HarmonySearch import harmony_search
from benchmark_functions.many_local_minimums import *
from benchmark_functions.bowl_shape import *
from benchmark_functions.gradients import *


def test_batch_gradient_descent():
    print(batch_gradient_descent(rotated_hyper_ellipsoid_gradient, 4, _range=130))  # works up to d=4
    print(batch_gradient_descent(sphere_gradient, 18))  # works perfectly with any parameters
    print(batch_gradient_descent(sum_of_powers_gradient, 4, _range=2))  # works well up to d=4, then precision falls
    print(batch_gradient_descent(sum_of_squares_gradient, 4, _range=20))  # works up to d=4
    print(batch_gradient_descent(trid_gradient, 10, _range=200))  # works perfectly with any parameters


def test_approximated_gradient_descent():
    print(approximated_gradient_descent(rotated_hyper_ellipsoid, 4, _range=130))  # works up to d=4
    print(approximated_gradient_descent(sphere, 18))  # works perfectly with any parameters
    print(approximated_gradient_descent(sum_of_powers, 2, _range=2))  # works well up to d=1, then precision falls
    print(approximated_gradient_descent(sum_of_squares, 4, _range=20))  # works up to d=4
    print(approximated_gradient_descent(trid, 10, _range=200))  # works well with any parameters


def test_hs():
    print(harmony_search(ackley, 10))
    print(harmony_search(griewank, 10))
    print(harmony_search(levy, 10))
    print(harmony_search(rastrigin, 10))
    print(harmony_search(schwefel, 10))
    print(harmony_search(rotated_hyper_ellipsoid, 10, _range=130))
    print(harmony_search(sphere, 10))
    print(harmony_search(sum_of_powers, 10, _range=2))
    print(harmony_search(sum_of_squares, 10, _range=20))


test_batch_gradient_descent()
test_approximated_gradient_descent()
test_hs()
