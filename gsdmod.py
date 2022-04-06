import numpy as np

from function import grad, f


class GradientMod(object):
    def __init__(self, x, y, learning_rate):
        self.y = y
        self.x = x
        self.learning_rate = learning_rate

    def gradient(self, current_point, previous_grad):
        return np.array(grad(self.x, self.y, current_point))


class MomentumGradientMod(GradientMod):
    def __init__(self, x, y, learning_rate, beta):
        super().__init__(x, y, learning_rate)
        self.beta = beta

    def mod(self, point, previous_grad):
        return self.beta * previous_grad + (1 - self.beta) * self.gradient(point, previous_grad)


class NesterovGradientMod(GradientMod):
    def __init__(self, x, y, learning_rate, beta):
        super().__init__(x, y, learning_rate)
        self.beta = beta

    def gradient(self, point, previous_grad):
        return np.array(grad(self.x, self.y, point - self.beta * previous_grad))

    def mod(self, point, previous_grad):
        return self.beta * previous_grad + self.learning_rate * self.gradient(point, previous_grad)


