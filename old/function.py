import numpy as np
from numpy import random
from scipy import optimize
from deprecated import deprecated

min_point = ()

f_counter = 0


@deprecated
def f(*args):
    global f_counter

    f_counter += 1

    return random_quad(*args)


@deprecated
def cubic(x, y): return 7 * (x - 1) ** 2 + 3 * (y + 8) ** 2 + 3 * x * y


@deprecated
def cubic2(x, y): return 2 * (x + 5) ** 2 + 4 * (y + 3) ** 2


random_seed = 123412

dim = 3


@deprecated
def set_dim(another):
    global dim

    dim = another


@deprecated
def get_dim():
    return dim


@deprecated
def random_quad(*arg_array):
    accumulator = 0
    mul_k = 9
    add_k = 15
    np.random.seed(123)
    for arg in arg_array:
        accumulator += int(random.uniform(1, mul_k)) * (arg + random.uniform(-add_k, add_k)) ** 2
    return accumulator


@deprecated
def get_f_counter():
    return f_counter


grad_counter = 0


@deprecated
def get_grad_counter():
    return grad_counter


@deprecated
def grad(point):
    global grad_counter
    grad_counter += 1
    h = 1e-5
    result = np.zeros(point.size)
    for i, n in enumerate(point):
        point[i] = point[i] + h
        f_plus = f(*point)
        point[i] = point[i] - 2 * h
        f_minus = f(*point)
        point[i] = point[i] + h
        result[i] = (f_plus - f_minus) / (2 * h)
    return result


@deprecated
def find_min_point():
    global min_point
    best_x = optimize.minimize(f, np.array(-3), args=(-3,)).x
    best_y = optimize.minimize(lambda x, y: f(y, x), np.array(-3), args=(-3,)).x
    min_point = best_x[0], best_y[0]


@deprecated
def get_min_point():
    return min_point
