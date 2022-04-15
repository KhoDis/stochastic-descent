import os

import numpy as np

from matplotlib import pyplot as plt, cm

dtype_ = np.dtype("float64")


class SGDResult:
    def __init__(self, rescaled_scalars, scaled_scalars, func_utils, scaler_name, method_name):
        self.rescaled_scalars = rescaled_scalars
        self.scaled_scalars = scaled_scalars
        self.func_utils = func_utils
        self.scaler_name = scaler_name
        self.method_name = method_name


class SurfaceConfig(object):
    def __init__(self, alpha, color):
        self.alpha = alpha
        self.color = color


class ContourConfig(object):
    def __init__(self, cmap, alpha, linestyles, linewidths):
        self.cmap = cmap
        self.alpha = alpha
        self.linestyles = linestyles
        self.linewidths = linewidths


class LineConfig(object):
    def __init__(self, cmap, alpha, marker, markersize, linestyle, linewidth):
        self.cmap = cmap
        self.alpha = alpha
        self.linestyle = linestyle
        self.linewidth = linewidth
        self.marker = marker
        self.markersize = markersize


class Drawer(object):
    def __init__(self, sgd_result):
        self.points = np.array(sgd_result.scaled_scalars, dtype_)
        self.func_utils = sgd_result.func_utils
        self.X, self.Y, self.Z = self.__calculate_values(shift=10, amount=1000)
        self.scaler_name = sgd_result.scaler_name
        self.method_name = sgd_result.method_name
        self.surface_config = SurfaceConfig(alpha=0.2, color='green')
        self.line_config = LineConfig(
            lambda lightness: cm.hot(lightness),
            alpha=1, marker='o', markersize=3, linestyle='solid', linewidth=2
        )
        self.contour_config = ContourConfig(cmap='autumn', alpha=0.8, linestyles='dashed', linewidths=0.5)

    def __calculate_values(self, shift, amount):
        last_point = self.points[-1]
        X = np.linspace(last_point[0] - shift, last_point[0] + shift, amount)
        Y = np.linspace(last_point[1] - shift, last_point[1] + shift, amount)

        Z = np.ndarray((amount, amount), dtype_)
        for i in range(0, len(X)):
            for j in range(0, len(Y)):
                Z[j][i] = self.func_utils.f([X[i], Y[j]])

        return X, Y, Z

    def draw_3d(self, show_image=True):
        ax = plt.figure(figsize=(5, 5)).add_subplot(111, projection='3d')
        ax.plot_surface(*np.meshgrid(self.X, self.Y), self.Z,
                        color=self.surface_config.color,
                        alpha=self.surface_config.alpha)

        colors = iter(self.line_config.cmap(np.flip(np.linspace(0, 1, len(self.points)))))
        for i in range(1, len(self.points)):
            color = next(colors)
            ax.plot([self.points[i - 1][0], self.points[i][0]],
                    [self.points[i - 1][1], self.points[i][1]],
                    [self.func_utils.f(self.points[i - 1]), self.func_utils.f(self.points[i])],
                    alpha=self.line_config.alpha,
                    color=color,
                    marker=self.line_config.marker,
                    markersize=self.line_config.markersize,
                    linestyle=self.line_config.linestyle,
                    linewidth=self.line_config.linewidth)

        ax.contour(self.X, self.Y, self.Z, sorted([self.func_utils.f(p) for p in self.points]),
                   alpha=self.contour_config.alpha,
                   cmap=self.contour_config.cmap,
                   linestyles=self.contour_config.linestyles,
                   linewidths=self.contour_config.linewidths)

        if show_image:
            plt.show()

        self.__save_plot('./img/3d')
        plt.close()

    def draw_2d(self, show_image=True):
        plt.title('2d projection')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')

        plt.plot(self.points[:, 0], self.points[:, 1],
                 marker=self.line_config.marker,
                 linestyle=self.line_config.linestyle,
                 linewidth=self.line_config.linewidth)
        plt.text(*self.points[0], self.__point_text(self.points[0]))
        plt.text(*self.points[-1], self.__point_text(self.points[-1]))
        plt.contour(self.X, self.Y, self.Z, sorted([self.func_utils.f(p) for p in self.points]))

        if show_image:
            plt.show()

        self.__save_plot('./img/2d')
        plt.close()

    def __point_text(self, point):
        return f'({point[0]:.2f}, {point[1]:.2f}) - {self.func_utils.f(point):.5f}'

    def __save_plot(self, directory):
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{directory}/scaler-{self.scaler_name}_method-{self.method_name}.png')
