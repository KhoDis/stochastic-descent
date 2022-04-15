import numpy as np
from matplotlib import pyplot as plt, cm


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
        self.direction = -func_utils.gradient(point)

        X = np.linspace(-6, 6, 30)
        Y = np.linspace(-6, 6, 30)
        Z = np.ndarray((30, 30), np.dtype("float64"))
        for i in range(0, len(X)):
            for j in range(0, len(Y)):
                Z[j][i] = self.func([X[i], Y[j]])

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.contour3D(X, Y, Z, 50, cmap='binary')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.scatter(self.point[0], self.point[1], self.func(self.point), color='red', marker='o')
        new_point = self.point + 1 * self.direction
        ax.plot(xs=[self.point[0], new_point[0]],
                ys=[self.point[1], new_point[1]],
                zs=[self.func(self.point)[0], self.func(new_point)[0]])
        plt.show()

    def g(self, alpha):
        # print(self.point, alpha, self.direction)
        return self.func(self.point + alpha * self.direction)

    def calculate(self, eps=0.0001, delta=0.01, strength=2):
        alpha_beg = 0
        alpha_mid = 0.01
        alpha_end = self.__generate_next_alpha(alpha_mid, strength)

        print(alpha_beg, alpha_mid)
        return self.__dichotomy(alpha_beg, alpha_end, eps, delta)

    def __dichotomy(self, begin, end, eps, delta):
        while abs(end - begin) / 2 > eps:
            x1 = (begin + end) / 2 - delta
            x2 = (begin + end) / 2 + delta

            g_x1 = self.g(x1)
            g_x2 = self.g(x2)

            if g_x1 <= g_x2:
                end = x2
            elif g_x1 > g_x2:
                begin = x1

            # print('point', begin, end)

        return (begin + end) / 2

    def __generate_next_alpha(self, alpha_curr, strength):
        g_curr = self.g(alpha_curr)

        alpha = alpha_curr * strength
        g_next = self.g(alpha)

        while g_curr >= g_next:
            alpha *= strength
            print(alpha)
            g_next = self.g(alpha)

        return alpha
