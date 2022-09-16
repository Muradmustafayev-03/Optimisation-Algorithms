from iterative_algorithms.gradient_descent import gradient_descent
from benchmark_functions.gradients import sphere_gradient

print(gradient_descent(sphere_gradient, 10))
