from iterative_algorithms.GradientDescent import batch_gradient_descent, approximated_gradient_descent
from metaheuristic_algorithms.HarmonySearch import HarmonySearch
from metaheuristic_algorithms.GeneticAlgorithm import GeneticAlgorithm
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
    print(HarmonySearch(ackley, 10).find_harmony())
    print(HarmonySearch(griewank, 10).find_harmony())
    print(HarmonySearch(levy, 10).find_harmony())
    print(HarmonySearch(rastrigin, 10).find_harmony())
    print(HarmonySearch(schwefel, 10).find_harmony())
    print(HarmonySearch(rotated_hyper_ellipsoid, 10, _range=130).find_harmony())
    print(HarmonySearch(sphere, 10).find_harmony())
    print(HarmonySearch(sum_of_powers, 10, _range=2).find_harmony())
    print(HarmonySearch(sum_of_squares, 10, _range=20).find_harmony())


def test_ga():
    print(GeneticAlgorithm(ackley, 10).evolve())
    print(GeneticAlgorithm(griewank, 10).evolve())
    print(GeneticAlgorithm(levy, 10).evolve())
    print(GeneticAlgorithm(rastrigin, 10).evolve())
    print(GeneticAlgorithm(schwefel, 10).evolve())
    print(GeneticAlgorithm(rotated_hyper_ellipsoid, 10).evolve())
    print(GeneticAlgorithm(sphere, 10).evolve())
    print(GeneticAlgorithm(sum_of_powers, 10).evolve())
    print(GeneticAlgorithm(sum_of_squares, 10).evolve())


test_hs()
