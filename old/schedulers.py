from numpy import exp

from old.search_algorithms import Dichotomy


class ConstLRScheduler:
    def __init__(self, const):
        self.const = const

    # noinspection PyUnusedLocal
    def show(self, n, func_utils, point):
        return self.const

    @staticmethod
    def name():
        return 'const'


class SteppedLRScheduler:
    def __init__(self, start, step):
        self.start = start
        self.step = step

    # noinspection PyUnusedLocal
    def show(self, n, func_utils, point):
        return self.start + self.step * n

    @staticmethod
    def name():
        return 'stepped'


class ExponentialLRScheduler:
    def __init__(self, start, step):
        self.start = start
        self.step = step

    # noinspection PyUnusedLocal
    def show(self, n, func_utils, point):
        return self.start * exp(-n * self.step)

    @staticmethod
    def name():
        return 'exponential'


class DichotomyScheduler:
    # noinspection PyUnusedLocal
    @staticmethod
    def show(n, func_utils, point):
        return Dichotomy(point, func_utils).calculate()

    @staticmethod
    def name():
        return 'dichotomy'
