import os
import sys
import numpy as np

from matplotlib import pyplot as plt, cm

dtype_ = np.dtype("float64")


class SGDResult:
    def __init__(self,
                 rescaled_scalars, scaled_scalars,
                 func_utils, batch_size,
                 scaler_name, method_name,
                 time, memory, grad_calls):
        self.rescaled_scalars = rescaled_scalars
        self.scaled_scalars = scaled_scalars
        self.func_utils = func_utils
        self.batch_size = batch_size
        self.scaler_name = scaler_name
        self.method_name = method_name
        self.time = time
        self.memory = memory
        self.grad_calls = grad_calls


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
        self.scaled = np.array(sgd_result.scaled_scalars, dtype_)
        self.rescaled = np.array(sgd_result.rescaled_scalars, dtype_)
        self.func_utils = sgd_result.func_utils
        self.X, self.Y, self.Z = self.__calculate_values(amount=1000)
        self.batch_size = sgd_result.batch_size
        self.scaler_name = sgd_result.scaler_name
        self.method_name = sgd_result.method_name
        self.surface_config = SurfaceConfig(alpha=0.2, color='green')
        self.line_config = LineConfig(
            lambda lightness: cm.hot(lightness),
            alpha=1, marker='o', markersize=3, linestyle='solid', linewidth=2
        )
        self.contour_config = ContourConfig(cmap='autumn', alpha=0.8, linestyles='dashed', linewidths=0.5)

    def __calculate_values(self, amount):
        last_point = self.scaled[-1]
        start_point = self.scaled[0]

        x_shift = abs(last_point[0] - start_point[0]) + 5
        y_shift = abs(last_point[1] - start_point[1]) + 5

        x = np.linspace(
            last_point[0] - x_shift,
            last_point[0] + x_shift,
            amount
        )

        y = np.linspace(
            last_point[1] - y_shift,
            last_point[1] + y_shift,
            amount
        )

        z = np.ndarray((amount, amount), dtype_)
        for i in range(0, len(x)):
            for j in range(0, len(y)):
                z[j][i] = self.func_utils.f([x[i], y[j]])

        return x, y, z

    def draw_3d(self, show_image=True):
        ax = plt.figure(figsize=(5, 5)).add_subplot(111, projection='3d')
        ax.plot_surface(*np.meshgrid(self.X, self.Y), self.Z,
                        color=self.surface_config.color,
                        alpha=self.surface_config.alpha)

        colors = iter(self.line_config.cmap(np.flip(np.linspace(0, 1, len(self.scaled)))))
        for i in range(1, len(self.scaled)):
            color = next(colors)
            ax.plot([self.scaled[i - 1][0], self.scaled[i][0]],
                    [self.scaled[i - 1][1], self.scaled[i][1]],
                    [self.func_utils.f(self.scaled[i - 1]), self.func_utils.f(self.scaled[i])],
                    alpha=self.line_config.alpha,
                    color=color,
                    marker=self.line_config.marker,
                    markersize=self.line_config.markersize,
                    linestyle=self.line_config.linestyle,
                    linewidth=self.line_config.linewidth)

        ax.contour(self.X, self.Y, self.Z, sorted([self.func_utils.f(p) for p in self.scaled]),
                   alpha=self.contour_config.alpha,
                   cmap=self.contour_config.cmap,
                   linestyles=self.contour_config.linestyles,
                   linewidths=self.contour_config.linewidths)

        self.__complete_plot('./img/3d', show_image)

    def draw_2d(self, show_image=True):
        plt.title('2d-b1 projection')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')

        plt.text(*self.scaled[0], self.__point_text(self.scaled[0]))
        plt.text(*self.scaled[-1], self.__point_text(self.scaled[-1]))
        plt.contour(self.X, self.Y, self.Z, sorted([self.func_utils.f(p) for p in self.scaled]))
        plt.plot(self.scaled[:, 0], self.scaled[:, 1],
                 marker=self.line_config.marker,
                 linestyle=self.line_config.linestyle,
                 linewidth=self.line_config.linewidth)

        self.__complete_plot('./img/2d', show_image)

    def draw_linear_regression(self, x, y, nth, show_image=True):
        scalars = self.rescaled[0::nth]

        shift = 2

        plt.title('Linear Regression')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')

        ax = plt.gca()
        ax.set_xlim([np.amin(x) - shift, np.amax(x) + shift])
        ax.set_ylim([np.amin(y) - shift, np.amax(y) + shift])

        for point_index in range(0, len(x)):
            plt.scatter(x[point_index][0], y[point_index], c="green")

        color = iter(cm.coolwarm(np.linspace(0, 1, len(scalars))))
        for id, scalar in enumerate(scalars):
            c = next(color)
            x = np.linspace(np.amin(x) - shift, np.amax(x) + shift, 100)
            y = scalar[0]
            for i in range(1, len(scalar)):
                y += x * scalar[i]

            linewidth = 3 if id == 0 or id == (len(scalars) - 1) else 0.6
            plt.plot(x, y, color=c, linewidth=linewidth)

        self.__complete_plot('./img/linear_regression', show_image)

    def __point_text(self, point):
        return f'({point[0]:.2f}, {point[1]:.2f}) - {self.func_utils.f(point):.5f}'

    def __complete_plot(self, directory, show_image):
        os.makedirs(directory, exist_ok=True)
        plt.savefig(f'{directory}/scaler-{self.scaler_name}_method-{self.method_name}_batch-{self.batch_size}.png')

        if show_image:
            plt.show()

        plt.close()

