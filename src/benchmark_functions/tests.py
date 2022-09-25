from bowl_shape import *
from many_local_minimums import *
from other import *

v0 = [[0 for _ in range(1, d+1)] for d in range(1, 11)]
v1 = [[1 for _ in range(1, d+1)] for d in range(1, 11)]

v_schwefel_min = [[420.9687 for _ in range(1, d+1)] for d in range(1, 11)]
v_styblinski_tang_min = [[-2.903534 for _ in range(1, d+1)] for d in range(1, 11)]
v_michalewicz_min = [2.20, 1.57]

v_dixon_min = [[2 ** (-1 * ((2 ** i - 2) / 2 ** i)) for i in range(1, d+1)] for d in range(1, 11)]
v_perm_min = [[i for i in range(1, d+1)] for d in range(1, 11)]
v_perm0_min = [[1/i for i in range(1, d+1)] for d in range(1, 11)]
v_trid_min = [[i * (d + 1 - i) for i in range(1, d+1)] for d in range(1, 11)]


def test_ackley():
    try:
        for vector in v0:
            assert round(ackley(vector), 15) == 0
        print('Ackley function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Ackley function does not work correctly at points (0,...,0)')
        return False


def test_dixon():
    try:
        for vector in v_dixon_min:
            assert round(dixon_price(vector), 30) == 0
        print('Dixon Price function works correctly at points x_i = 2^(-(2^i - 2)/2^i) for i = 1,...,d')
        return True
    except AssertionError:
        print('Dixon Price function does not work correctly at points '
              'x_i = 2^(-(2^i - 2)/2^i) for i = 1,...,d')
        return False


def test_griewank():
    try:
        for vector in v0:
            assert griewank(vector) == 0
        print('Griewank function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Griewank function does not work correctly at points (0,...,0)')
        return False


def test_levy():
    try:
        for vector in v1:
            assert round(levy(vector), 31) == 0
        print('Levy function works correctly at points (1,...,1)')
        return True
    except AssertionError:
        print('Levy function does not work correctly at points (1,...,1)')
        return False


def test_michalewicz():
    try:
        assert round(michalewicz(v_michalewicz_min), 1) == -1.8
        print('Michalewicz function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Michalewicz does not function work correctly at points (0,...,0)')
        return False


def test_perm():
    try:
        for vector in v_perm_min:
            assert round(perm(vector), 31) == 0
        print('Perm (d, b) function works correctly at points (1,2,...,d)')
        return True
    except AssertionError:
        print('Perm (d, b) function does not work correctly at points (1,2,...,d)')
        return False


def test_perm0():
    try:
        for vector in v_perm0_min:
            assert round(perm0(vector), 31) == 0
        print('Perm (0, d, b) function works correctly at points (1, 1/2,...,1/d)')
        return True
    except AssertionError:
        print('Perm (0, d, b) function does not work correctly at points (1, 1/2,...,1/d)')
        return False


def test_rastrigin():
    try:
        for vector in v0:
            assert rastrigin(vector) == 0
        print('Rastrigin function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Rastrigin function does not work correctly at points (0,...,0)')
        return False


def test_rosenbrock():
    try:
        for vector in v1:
            assert rosenbrock(vector) == 0
        print('Rosenbrock function works correctly at points (1,...,1)')
        return True
    except AssertionError:
        print('Rosenbrock function does not work correctly at points (1,...,1)')
        return False


def test_rotated_hyper_ellipsoid():
    try:
        for vector in v0:
            assert rotated_hyper_ellipsoid(vector) == 0
        print('Rotated Hyper Ellipsoid function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Rotated Hyper Ellipsoid function does not work correctly at points (0,...,0)')
        return False


def test_schwefel():
    try:
        for vector in v_schwefel_min:
            assert round(schwefel(vector), 3) == 0
        print('Schwefel function works correctly at points (420.9687,...,420.9687)')
        return True
    except AssertionError:
        print('Schwefel function does not work correctly at points (420.9687,...,420.9687)')
        return False


def test_sphere():
    try:
        for vector in v0:
            assert sphere(vector) == 0
        print('Sphere function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Sphere function does not work correctly at points (0,...,0)')
        return False


def test_styblinski_tang():
    try:
        for vector in v_styblinski_tang_min:
            assert round(styblinski_tang(vector), 1) == round(-39.16599 * len(vector), 1)
        print('Styblinski Tang function works correctly at points (2.903534,...,2.903534)')
        return True
    except AssertionError:
        print('Styblinski Tang function does not work correctly at points (2.903534,...,2.903534)')
        return False


def test_sum_of_powers():
    try:
        for vector in v0:
            assert sum_of_powers(vector) == 0
        print('Sum of Powers function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Sum of Powers function does not work correctly at points (0,...,0)')
        return False


def test_sum_of_squares():
    try:
        for vector in v0:
            assert sum_of_squares(vector) == 0
        print('Sum of Squares function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Sum of Squares function does not work correctly at points (0,...,0)')
        return False


def test_trid():
    try:
        for vector in v_trid_min:
            d = len(vector)
            assert trid(vector) == -d * (d+4) * (d-1) / 6
        print('Trid function works correctly at points x_i = i(d+1-i), for all i in 1,2,...,d')
        return True
    except AssertionError:
        print('Trid function does not work correctly at points x_i = i(d+1-i), for all i in 1,2,...,d')
        return False


def test_zakharov():
    try:
        for vector in v0:
            assert zakharov(vector) == 0
        print('Zakharov function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Zakharov function does not work correctly at points (0,...,0)')
        return False


if __name__ == '__main__':
    test_ackley()
    test_dixon()
    test_griewank()
    test_levy()
    test_michalewicz()
    test_perm()
    test_perm0()
    test_rastrigin()
    test_rosenbrock()
    test_rotated_hyper_ellipsoid()
    test_schwefel()
    test_sphere()
    test_styblinski_tang()
    test_sum_of_powers()
    test_sum_of_squares()
    test_trid()
    test_zakharov()
