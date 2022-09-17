from gradient_algorithms.GradientDescent import gradient_descent
from metaheuristic_algorithms.HarmonySearch import harmony_search
from benchmark_functions.gradients import *
from benchmark_functions.many_local_minimums import *


def test_gradient_descent():
    print(gradient_descent(rotated_hyper_ellipsoid_gradient, 4, _range=130))  # works up to d=4
    print(gradient_descent(sphere_gradient, 18))  # works perfectly with any parameters
    print(gradient_descent(sum_of_powers_gradient, 10, _range=2))  # works well up to d=4, for d>4 precision is less
    print(gradient_descent(sum_of_squares_gradient, 4, _range=20))  # works up to d=4
    print(gradient_descent(trid_gradient, 10, _range=200))  # works perfectly with any parameters


def test_hs():
    print(harmony_search(ackley))
    print(harmony_search(griewank))
    print(harmony_search(levy))
    print(harmony_search(rastrigin))
    print(harmony_search(schwefel))


test_gradient_descent()
test_hs()

