import os
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt, cm

from old.search_algorithms import Dichotomy
from dataset_reader import DatasetReader
from function import gradient, f
from abstractscaler import *
from gsdmod import DefaultGradientMod, MomentumGradientMod

mpl.use('TkAgg')

dtype_ = np.dtype("float64")


def draw_2d_surface(points, x, y, scaler_name, method_name):
    points = np.array(points, dtype_)

    shift = 1
    last_point = points[-1]
    x_axis = np.linspace(last_point[0] - shift, last_point[0] + shift, 1000)
    y_axis = np.linspace(last_point[1] - shift, last_point[1] + shift, 1000)

    zp = np.ndarray((1000, 1000))
    for i in range(0, len(x_axis)):
        for j in range(0, len(y_axis)):
            zp[j][i] = f([x_axis[i], y_axis[j]], x, y)
    # grid = np.meshgrid(x_axis, y_axis)

    plt.plot(points[:, 0], points[:, 1], 'o-')
    plt.text(*points[0], f'({points[0][0]}, {points[0][1]}) - {f(points[0], x, y):.5f}')
    plt.text(*points[-1], f'({points[-1][0]:.2f}, {points[-1][1]:.2f}) - {f(points[-1], x, y):.5f}')
    plt.contour(x_axis, y_axis, zp, sorted([f(p, x, y) for p in points]))
    # contours = plt.contour(*grid, f(grid, x, y), levels=levels)

    # TODO: sort out
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(*np.meshgrid(x_axis, y_axis), zp)

    path = f'./img/2d_surface/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(f'{path}/scaler-{scaler_name}_method-{method_name}.png')

    # plt.show()
    plt.clf()


def scale(axis):
    axis_min = np.amin(axis)
    axis_max = np.amax(axis)
    scaled = (axis - axis_min) / (axis_max - axis_min)
    return scaled, [axis_min, axis_max - axis_min]


def sgd(x, y, start, batch_size=1, epoch=50, tolerance=1e-06, random_state=None,
        scaler_ctor=None,
        sgd_mod=DefaultGradientMod(0.1)):
    scaler_ctor = DefaultScaler if scaler_ctor is None else scaler_ctor


    x = np.array(x, dtype=dtype_)
    y = np.array(y, dtype=dtype_)
    n_obs = len(x)

    scaler = scaler_ctor(x)
    is_scaled = scaler is not DefaultScaler
    x = scaler.data

    method_name = type(sgd_mod).__name__[:-3]
    scaler_name = type(scaler).__name__

    # Initializing the random number generator for shuffling
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    # Setting up and checking the learning rate
    # learn_rate = np.array(learn_rate, dtype=dtype_)
    # if np.any(learn_rate <= 0):
    #     raise ValueError("'learn_rate' must be greater than zero")

    batch_size = int(batch_size)
    if not 0 < batch_size <= n_obs:
        raise ValueError("'batch_size' must be greater than zero and less than or equal to the number of observations")

    epoch = int(epoch)
    if epoch <= 0:
        raise ValueError("'epoch' must be greater than zero")

    # Setting up and checking the tolerance
    # We need it to check if the absolute difference is small enough
    # We want to ignore absurdly different points to not to break our predictions
    # tolerance = np.array(tolerance, dtype=dtype_)
    # if np.any(tolerance <= 0):
    #     raise ValueError("'tolerance' must be greater than zero")

    current_point = np.array(start, dtype=dtype_)
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    beta = 0.9
    learning_rate = 0.1
    previous_info = np.zeros(current_point.shape[0])

    scalars = [start]
    for i in range(epoch):
        rng.shuffle(xy)

        for start in range(0, n_obs, batch_size):
            stop = start + batch_size
            x_batch = xy[start:stop, :-1]
            y_batch = xy[start:stop, -1:]

            # TODO: УРОД НЕ УДАЛЯЙ ТУДУШКИ БЕЗ МОЕГО ВЕДОМА СВИН. ЭТО МОЖНО ПЕРЕПИСАТЬ ПРОСТО В МЕТОД
            # NOTE: Working variant
            grad_ = np.array(gradient(x_batch, y_batch, current_point), dtype_)
            learning_rate = Dichotomy(current_point, -grad_, xy[start:stop, :-1], xy[start:stop, -1:]).calculate()
            diff = -learning_rate * grad_

            # direct = sgd_mod.direction(x_batch, y_batch, current_point)
            # diff, previous_info = momentum(x_batch, y_batch, previous_info, current_point, beta, learning_rate)

            # There's no point to iterate further if we found the minima
            # if np.all(np.abs(diff) <= tolerance):
            #     break

            current_point += diff
            scalars.append(current_point.copy())

    draw_2d_surface(scalars, x, y, scaler_name, method_name)
    return scaler.rescale(scalars)


def momentum(x, y, point, previous_grad, beta=0.9, learning_rate=0.1):
    grad = np.array(gradient(x, y, point), dtype_)
    new_grad = beta * previous_grad + (1 - beta) * grad
    direction = - learning_rate * new_grad
    return direction, grad


def nesterov(x, y, point, previous_grad, beta=0.9, learning_rate=0.1):
    grad = np.array(gradient(x, y, point - beta * previous_grad), dtype_)
    new_grad = beta * previous_grad + learning_rate * grad
    direction = -grad
    return direction, new_grad


def rms(x, y, point, previous_v, beta=0.9, learning_rate=0.1):
    grad = np.array(gradient(x, y, point), dtype_)
    new_v = beta * previous_v + (1 - beta) * grad ** 2
    direction = - learning_rate / np.sqrt(new_v) * grad
    return direction, new_v


def ada(x, y, point, previous_s, beta=0.9, learning_rate=0.1):
    grad = np.array(gradient(x, y, point), dtype_)
    new_s = previous_s + grad ** 2
    direction = - learning_rate / np.sqrt(new_s) * grad
    return direction, new_s


def main():
    x, y = DatasetReader('planar').data
    # x_batch = [[1, 2, 3], [1, 2, 3], ..., ] # n-1 мерная точка
    # y_batch = [1, 2, ..., ]
    scalars = sgd(x, y, start=[0, 1], batch_size=5, epoch=20, random_state=0,
                  scaler_ctor=MinMaxScaler,
                  sgd_mod=MomentumGradientMod(0.1, 0.9))

    print("Optimal:", scalars[-1])
    # draw_linear_regression(scalars, x, y)


def draw_linear_regression(scalars, x, y):
    plt.title('Gradient Descent')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    for point_index in range(0, len(x)):
        plt.scatter(x[point_index][0], y[point_index], c="green")
    color = iter(cm.rainbow(np.linspace(0, 1, len(scalars))))
    for scalar in scalars:
        c = next(color)
        x = np.linspace(0, np.amax(x), 100)
        y = scalar[0]
        for i in range(1, len(scalar)):
            y += x * scalar[i]
        plt.plot(x, y, color=c)
    plt.show()


if __name__ == '__main__':
    main()
