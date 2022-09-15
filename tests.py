from iterative_algorithms.gradient_descent import gradient_descent
from benchmark_functions.gradients import perm0_gradient

print(gradient_descent(perm0_gradient, 3))
