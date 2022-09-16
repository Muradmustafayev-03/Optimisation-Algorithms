from iterative_algorithms.gradient_descent import gradient_descent
from benchmark_functions.gradients import sphere_gradient, sum_of_powers_gradient, sum_of_squares_gradient, \
    trid_gradient

print(gradient_descent(sphere_gradient, 10))
print(gradient_descent(sum_of_powers_gradient, 4, _range=3))
print(gradient_descent(sum_of_squares_gradient, 4, _range=20))
print(gradient_descent(trid_gradient, 5, _range=50))

