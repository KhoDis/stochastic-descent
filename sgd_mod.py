import numpy as np


class DefaultGradientMod(object):
    def _gradient(self, f, point):
        return np.array(f.gradient(point))

    def diff(self, f, point, learning_rate):
        return -learning_rate * self._gradient(f, point)


class MomentumGradientMod(DefaultGradientMod):
    def __init__(self, beta):
        self.beta = beta
        self.previous_grad = None

    def diff(self, f, point, learning_rate):
        self.previous_grad = np.zeros(point.shape[0]) if self.previous_grad is None else self.previous_grad

        new_grad = self.beta * self.previous_grad + (1 - self.beta) * self._gradient(f, point)

        self.previous_grad = new_grad

        return -learning_rate * new_grad


class NesterovGradientMod(DefaultGradientMod):
    def __init__(self, beta):
        self.beta = beta
        self.previous_grad = None

    def _gradient(self, f, point):
        return np.array(f.gradient(point - self.beta * self.previous_grad))

    def diff(self, f, point, learning_rate):
        self.previous_grad = np.zeros(point.shape[0]) if self.previous_grad is None else self.previous_grad

        new_grad = self.beta * self.previous_grad + learning_rate * self._gradient(f, point)

        self.previous_grad = new_grad

        return -new_grad


class AdaGradientMod(DefaultGradientMod):
    def __init__(self):
        self.previous_s = None

    def diff(self, f, point, learning_rate):
        self.previous_s = np.ones(point.shape[0]) if self.previous_s is None else self.previous_s

        grad = self._gradient(f, point)
        new_s = np.array(self.previous_s + grad ** 2)
        direction = - learning_rate / np.sqrt(new_s) * grad

        self.previous_s = new_s

        return direction


class RmsPropGradientMod(DefaultGradientMod):
    def __init__(self, beta):
        self.beta = beta
        self.previous_v = None

    def diff(self, f, point, learning_rate):
        self.previous_v = np.ones(point.shape[0]) if self.previous_v is None else self.previous_v

        grad = self._gradient(f, point)
        new_v = self.beta * self.previous_v + (1 - self.beta) * np.square(grad)
        direction = -learning_rate / np.sqrt(new_v) * grad

        self.previous_v = new_v
        return direction


class Adam(DefaultGradientMod):
    def __init__(self, beta1, beta2):
        self.beta1 = beta1
        self.beta2 = beta2
        self.previous_v = None
        self.previous_grad = None
        self.eps = 1e-8

    def diff(self, f, point, learning_rate):
        self.previous_v = np.ones(point.shape[0]) if self.previous_v is None else self.previous_v
        self.previous_grad = np.zeros(point.shape[0]) if self.previous_grad is None else self.previous_grad

        grad = self._gradient(f, point)
        new_grad = self.beta1 * self.previous_grad + (1 - self.beta1) * grad
        new_v = self.beta2 * self.previous_v + (1 - self.beta2) * np.square(grad)
        direction = -learning_rate * new_grad / (np.sqrt(new_v) + self.eps)

        self.previous_v = new_v
        self.previous_grad = new_grad

        return direction
