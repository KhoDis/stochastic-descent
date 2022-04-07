import numpy as np

from function import gradient, f


class DefaultGradientMod(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def __gradient(self, x, y, point):
        return np.array(gradient(x, y, point))

    def direction(self, x, y, point):
        return self.__gradient(x, y, point)


class MomentumGradientMod(DefaultGradientMod):
    def __init__(self, learning_rate, beta):
        super().__init__(learning_rate)
        self.beta = beta
        self.previous_grad = None  # TODO: we need to initialize this via np.array in order to make this work properly

    def direction(self, x, y, point):
        g = self.__gradient(x, y, point)

        grad = self.beta * self.previous_grad + (1 - self.beta) * g

        self.previous_grad = grad

        return grad


class NesterovGradientMod(DefaultGradientMod):
    def __init__(self, learning_rate, beta):
        super().__init__(learning_rate)
        self.beta = beta
        self.previous_grad = None  # TODO: we need to initialize this via np.array in order to make this work properly

    def __gradient(self, x, y, point):
        return np.array(gradient(x, y, point - self.beta * self.previous_grad))

    def direction(self, x, y, point):
        g = self.__gradient(x, y, point)

        grad = self.beta * self.previous_grad + self.learning_rate * g

        self.previous_grad = grad

        return grad


class AdaGradientMod(DefaultGradientMod):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)
        self.previous_s = None  # TODO: we need to initialize this via np.array in order to make this work properly
        self.previous_grad = 0  # TODO: we need to initialize this via np.array in order to make this work properly

    def direction(self, x, y, point):
        g = self.__gradient(x, y, point)
        s = np.array(self.previous_s + g ** 2)

        grad = -(self.learning_rate / np.sqrt(s)) * g

        self.previous_grad = grad
        self.previous_s = s

        return grad


class RmsProp(DefaultGradientMod):
    def __init__(self, learning_rate, beta):
        super().__init__(learning_rate)
        self.beta = beta
        self.previous_v = None  # TODO: we need to initialize this via np.array in order to make this work properly
        self.previous_grad = 0  # TODO: we need to initialize this via np.array in order to make this work properly

    def direction(self, x, y, point):
        g = self.__gradient(x, y, point)
        v = self.beta * self.previous_v + (1 - self.beta) * g ** 2

        grad = -(self.learning_rate / np.sqrt(v)) * g

        self.previous_grad = grad  # TODO: why it's not used
        self.previous_v = v

        return grad
