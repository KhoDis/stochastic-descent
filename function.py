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
