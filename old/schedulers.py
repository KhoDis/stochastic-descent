from deprecated import deprecated
from numpy import exp

from search_algorithms import Dichotomy


@deprecated
class ConstLRScheduler:
    def __init__(self, const):
        self.const = const

    # noinspection PyUnusedLocal
    def show(self, n, point, direction):
        return self.const

    @staticmethod
    def name():
        return 'const'


@deprecated
class SteppedLRScheduler:
    def __init__(self, start, step):
        self.start = start
        self.step = step

    # noinspection PyUnusedLocal
    def show(self, n, point, direction):
        return self.start + self.step * n

    @staticmethod
    def name():
        return 'stepped'


@deprecated
class ExponentialLRScheduler:
    def __init__(self, start, step):
        self.start = start
        self.step = step

    # noinspection PyUnusedLocal
    def show(self, n, point, direction):
        return self.start * exp(-n * self.step)

    @staticmethod
    def name():
        return 'exponential'


@deprecated
class DichotomyScheduler:
    # noinspection PyUnusedLocal
    # noinspection PyMethodMayBeStatic
    def show(self, n, point, direction):
        return Dichotomy(point, direction).calculate()

    @staticmethod
    def name():
        return 'dichotomy'
