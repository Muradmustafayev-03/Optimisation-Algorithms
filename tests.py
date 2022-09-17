from iterative_algorithms.gradient_descent import gradient_descent
from benchmark_functions.gradients import *

print(gradient_descent(rotated_hyper_ellipsoid_gradient, 4, _range=130))  # works up to d=4
print(gradient_descent(sphere_gradient, 18))  # works perfectly with any parameters
print(gradient_descent(sum_of_powers_gradient, 4, _range=3))  # works up to d=4
print(gradient_descent(sum_of_squares_gradient, 4, _range=20))  # works up to d=4
print(gradient_descent(trid_gradient, 10, _range=50))  # works perfectly with any parameters
