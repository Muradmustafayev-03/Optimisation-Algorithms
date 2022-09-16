from iterative_algorithms.gradient_descent import gradient_descent
from benchmark_functions.gradients import sphere_gradient, sum_of_powers_gradient

print(gradient_descent(sphere_gradient, 10))
print(gradient_descent(sum_of_powers_gradient, 3, _range=3,))
