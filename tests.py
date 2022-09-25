from iterative_algorithms.GradientDescent import BatchGradientDescent, ApproximatedGradientDescent
from metaheuristic_algorithms.HarmonySearch import HarmonySearch
from metaheuristic_algorithms.GeneticAlgorithm import GeneticAlgorithm
from benchmark_functions.many_local_minimums import *
from benchmark_functions.bowl_shape import *
from benchmark_functions.gradients import *


def test_batch_gradient_descent():
    print(BatchGradientDescent(rotated_hyper_ellipsoid_gradient, 4, _range=130).optimize())  # works up to d=4
    print(BatchGradientDescent(sphere_gradient, 18).optimize())  # works perfectly with any parameters
    print(BatchGradientDescent(sum_of_powers_gradient, 4, _range=2).optimize())  # works well up to d=4, then worse
    print(BatchGradientDescent(sum_of_squares_gradient, 4, _range=20).optimize())  # works up to d=4
    print(BatchGradientDescent(trid_gradient, 10, _range=200).optimize())  # works perfectly with any parameters


def test_approximated_gradient_descent():
    print(ApproximatedGradientDescent(rotated_hyper_ellipsoid, 4, _range=130).optimize())  # works up to d=4
    print(ApproximatedGradientDescent(sphere, 18).optimize())  # works perfectly with any parameters
    print(ApproximatedGradientDescent(sum_of_powers, 2, _range=2).optimize())  # works well up to d=2, then worse
    print(ApproximatedGradientDescent(sum_of_squares, 4, _range=20).optimize())  # works up to d=4
    print(ApproximatedGradientDescent(trid, 10, _range=200).optimize())  # works well with any parameters


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


test_batch_gradient_descent()
test_approximated_gradient_descent()
test_hs()
test_ga()
