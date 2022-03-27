import numpy as np


class Function(object):
    def __init__(self, f, dim):
        self._f = f
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def apply(self, args):
        return self._f(*args)

    def grad(self, point):
        h = 1e-5
        result = np.zeros(point.size)
        for i, n in enumerate(point):
            point[i] = point[i] + h
            f_plus = self._f(*point)
            point[i] = point[i] - 2 * h
            f_minus = self._f(*point)
            point[i] = point[i] + h
            result[i] = (f_plus - f_minus) / (2 * h)
        return result
