import os
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt, cm

from drawer import Drawer, SGDResult
from scaler import DefaultScaler, MinMaxScaler
from dataset_reader import DatasetReader
from func_utils import FuncUtils
# noinspection PyUnresolvedReferences
from sgd_mod import DefaultGradientMod, MomentumGradientMod, NesterovGradientMod, AdaGradientMod, RmsProp
# noinspection PyUnresolvedReferences
from old.schedulers import ConstLRScheduler, DichotomyScheduler

mpl.use('TkAgg')

# noinspection SpellCheckingInspection
dtype_ = np.dtype("float64")


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


def sgd(x, y, start, batch_size=1, epoch=50, random_state=None,
        scheduler=ConstLRScheduler(0.1),
        scaler_ctor=None,
        sgd_mod=DefaultGradientMod()):
    tolerance = 1e-06

    scaler_ctor = DefaultScaler if scaler_ctor is None else scaler_ctor

    x = np.array(x, dtype=dtype_)
    y = np.array(y, dtype=dtype_)
    n_obs = len(x)

    scaler = scaler_ctor(x)
    x = scaler.data

    method_name = type(sgd_mod).__name__[:-3]
    scaler_name = type(scaler).__name__

    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    batch_size = int(batch_size)
    if not 0 < batch_size <= n_obs:
        raise ValueError("'batch_size' must be greater than zero and less than or equal to the number of observations")

    epoch = int(epoch)
    if epoch <= 0:
        raise ValueError("'epoch' must be greater than zero")

    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    current_point = np.array(start, dtype=dtype_)
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    scalars = [start]
    iter_count = 0
    for i in range(epoch):
        rng.shuffle(xy)

        for start in range(0, n_obs, batch_size):
            iter_count += 1
            stop = start + batch_size
            x_batch = xy[start:stop, :-1]
            y_batch = xy[start:stop, -1:]

            u = FuncUtils(x_batch, y_batch)
            learning_rate = scheduler.show(iter_count, u, current_point)
            diff = sgd_mod.diff(u, current_point, learning_rate)

            if np.all(np.abs(diff) <= tolerance):
                break

            current_point += diff
            scalars.append(current_point.copy())

    return scaler.rescale(scalars), SGDResult(scalars, FuncUtils(x, y), scaler_name, method_name)


def main():
    x, y = DatasetReader('planar').data
    # x_batch = [[1, 2, 3], [1, 2, 3], ..., ] # n-1 мерная точка
    # y_batch = [1, 2, ..., ]
    scalars, sgd_result = sgd(x, y, start=[0, 1], batch_size=5, epoch=20, random_state=0,
                              scheduler=ConstLRScheduler(0.01),
                              scaler_ctor=MinMaxScaler,
                              sgd_mod=NesterovGradientMod(beta=0.9))

    drawer = Drawer(sgd_result)
    drawer.draw_2d(True)
    drawer.draw_3d(True)

    print("Optimal:", scalars[-1])
    # draw_linear_regression(scalars, x, y)


if __name__ == '__main__':
    main()
