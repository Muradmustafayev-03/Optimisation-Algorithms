from bowl_shape import *
from many_local_minimums import *
from other import *

v0 = [[0 for _ in range(1, d+1)] for d in range(1, 11)]
v1 = [[1 for _ in range(1, d+1)] for d in range(1, 11)]

v_schwefel_min = [[420.9687 for _ in range(1, d+1)] for d in range(1, 11)]
v_styblinski_tang_min = [[2.903534 for _ in range(1, d+1)] for d in range(1, 11)]
v_michalewicz_min = [2.20, 1.57]

v_dixon_min = [[2 ** (-1 * ((2 ** i - 2) / 2 ** i)) for i in range(1, d+1)] for d in range(1, 11)]
v_perm_min = [[i for i in range(1, d+1)] for d in range(1, 11)]
v_perm0_min = [[1/i for i in range(1, d+1)] for d in range(1, 11)]
v_trid_min = [[d+1-i for i in range(1, d+1)] for d in range(1, 11)]


def test_ackley():
    try:
        for vector in v0:
            assert round(ackley(vector), 15) == 0
        print('Ackley function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Ackley does not function work correctly at points (0,...,0)')
        return False


def test_griewank():
    try:
        for vector in v0:
            assert griewank(vector) == 0
        print('Griewank function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Griewank does not function work correctly at points (0,...,0)')
        return False


def test_levy():
    try:
        for vector in v1:
            assert round(levy(vector), 31) == 0
        print('Levy function works correctly at points (1,...,1)')
        return True
    except AssertionError:
        print('Levy does not function work correctly at points (1,...,1)')
        return False


def test_rastrigin():
    try:
        for vector in v0:
            assert rastrigin(vector) == 0
        print('Rastrigin function works correctly at points (0,...,0)')
        return True
    except AssertionError:
        print('Rastrigin does not function work correctly at points (0,...,0)')
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
