import numpy as np


def count_degree(number):
    count = 0
    while number > 0:
        number = number // 10
        count += 1
    return count - 2


class Dichotomy:
    def __init__(self, point, func_utils):
        self.point = point
        self.func = func_utils.f
        self.direction = -func_utils.f(point)

    def g(self, alpha):
        return self.func(self.point + alpha * self.direction)

    def calculate(self, eps=0.01, delta=0.01, strength=1.1):
        alpha_beg = 0
        alpha_mid = 10 ** (-count_degree(np.max(self.direction)))
        alpha_end = self.__generate_next_alpha(alpha_mid, strength)

        return self.__dichotomy(alpha_beg, alpha_end, eps, delta)

    def __dichotomy(self, begin, end, eps, delta):
        while abs(end - begin) / 2 > eps:
            x1 = (begin + end - delta) / 2
            x2 = (begin + end + delta) / 2

            g_x1 = self.g(x1)
            g_x2 = self.g(x2)

            if g_x1 <= g_x2:
                end = x2
            elif g_x1 > g_x2:
                begin = x1

        return (begin + end) / 2

    def __generate_next_alpha(self, alpha_curr, initial_value):
        g_curr = self.g(alpha_curr)

        alpha = initial_value
        g_next = self.g(alpha)

        while g_curr >= g_next:
            alpha *= alpha
            g_next = self.g(alpha)

        return alpha
