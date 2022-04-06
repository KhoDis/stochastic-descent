# class Function(object):
#     def __init__(self, f, dim):
#         self._f = f
#         self._dim = dim
#
#     @property
#     def dim(self):
#         return self._dim
#
#     def apply(self, args):
#         return self._f(*args)

import numpy as np


# noinspection DuplicatedCode
# TODO: remove DuplicateCode
def f(point, x, y):
    accumulator = 0
    for j in range(0, len(y)):
        prediction = point[0]
        for i in range(1, len(point)):
            prediction += x[j][i - 1] * point[i]
        accumulator += (y[j] - prediction) ** 2
    return accumulator


def grad(x, y, point):
    h = 1e-5
    result = np.zeros(point.size)
    for i, n in enumerate(point):
        point[i] = point[i] + h
        f_plus = f(point, x, y)
        point[i] = point[i] - 2 * h
        f_minus = f(point, x, y)
        point[i] = point[i] + h
        result[i] = (f_plus - f_minus) / (2 * h)
    return result
