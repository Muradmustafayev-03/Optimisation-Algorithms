from benchmark_functions import *
from metaheuristic_algorithms.HarmonySearch import harmony_search


def mean(arr):
    return sum(arr) / len(arr)


def stdev(arr):
    return sqrt(sum(square(x - mean(arr)) for x in arr) / 30)


ackley_res = []
for i in range(30):
    minimisation = harmony_search(func=ackley, _range=32.768)
    ackley_res.append(ackley(minimisation))

print("Mean minimum of Ackley function is " + str(mean(ackley_res)))
print("With the standard deviation of " + str(stdev(ackley_res)))
print("\n")


rastrigin_res = []
for i in range(30):
    minimisation = harmony_search(func=rastrigin)
    rastrigin_res.append(rastrigin(minimisation))

print("Mean minimum of Rastrigin function is " + str(mean(rastrigin_res)))
print("With the standard deviation of " + str(stdev(rastrigin_res)))
print("\n")


rosenbrock_res = []
for i in range(30):
    minimisation = harmony_search(func=rosenbrock)
    rosenbrock_res.append(rosenbrock(minimisation))

print("Mean minimum of Rosenbrock function is " + str(mean(rosenbrock_res)))
print("With the standard deviation of " + str(stdev(rosenbrock_res)))
print("\n")


schwefel_res = []
for i in range(30):
    minimisation = harmony_search(func=schwefel, par=0.3, bw=10)
    schwefel_res.append(schwefel(minimisation))

print("Mean minimum of Schwefel function is " + str(mean(schwefel_res)))
print("With the standard deviation of " + str(stdev(schwefel_res)))
print("\n")
