import timeit

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt, cm
from memory_profiler import memory_usage, profile

from dataset_reader import DatasetReader
from drawer import Drawer, SGDResult
from func_utils import FuncUtils
# noinspection PyUnresolvedReferences
from old.schedulers import ConstLRScheduler, DichotomyScheduler, ExponentialLRScheduler
from scaler import DefaultScaler, MinMaxScaler, MeanScaler, UnitLengthScaler
# noinspection PyUnresolvedReferences
from sgd_mod import *

mpl.use('TkAgg')

dtype_ = np.dtype("float64")


def sgd(x, y, start, batch_size=1, epoch=50, random_state=None,
        scheduler=None,
        scaler_ctor=None,
        sgd_mod=DefaultGradientMod()):
    tolerance = 1e-06

    scheduler = ConstLRScheduler(0.1) if scheduler is None else scheduler
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
    start = datetime.datetime.now()
    tracemalloc.start()
    for i in range(epoch):
        # rng.shuffle(xy)

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

    end = datetime.datetime.now()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return SGDResult(scaler.rescale(scalars), scalars, FuncUtils(x, y), batch_size, scaler_name, method_name, end - start, peak, get_counter())


def basic_sgd():
    x, y = DatasetReader('planar').data
    return sgd(x, y, start=[0, 1], batch_size=5, epoch=20, random_state=0,
               scheduler=ConstLRScheduler(0.01),
               scaler_ctor=MinMaxScaler,
               sgd_mod=NesterovGradientMod(beta=0.9))


def measure_memory():
    return memory_usage(basic_sgd)  # () is for args, {} is for kwargs


def main():
    x, y = DatasetReader('planar_hard').data
    # x_batch = [[1, 2, 3], [1, 2, 3], ..., ] # n-1 мерная точка
    # y_batch = [1, 2, ..., ]

    # sgd_result = sgd(x, y, start=[0, 1], batch_size=15, epoch=1000, random_state=0,
    #                  scheduler=ExponentialLRScheduler(1, 0.9),
    #                  scaler_ctor=DefaultScaler,
    #                  sgd_mod=DefaultGradientMod())

    # print('time: ', timeit.Timer('basic_sgd()', "from __main__ import basic_sgd").timeit(1))

    line_count = 50
    sgd_results = []
    for batch in [
        1,
        5,
        10,
        15
    ]:
        for scaler in [MinMaxScaler]:
            for mod, lr, epoch in [
                (DefaultGradientMod(), 0.04, 50),
                (NesterovGradientMod(beta=0.9), 0.006, 50),
                (MomentumGradientMod(beta=0.9), 0.1, 50),
                (AdaGradientMod(), 5, 50),
                (RmsPropGradientMod(beta=0.9), 0.8, 50),
                (AdamGradientMod(beta1=0.9, beta2=0.999), 0.25, 50),
            ]:
                sgd_result = task_3(batch_size=batch, scaler=scaler, mod=mod, x=x, y=y, lr=lr, epoch=epoch)
                sgd_results.append(sgd_result)

                drawer = Drawer(sgd_result)
                drawer.draw_2d(show_image=True)
                # drawer.draw_3d(show_image=True)

                print("Optimal:", sgd_result.rescaled_scalars[-1])
                nth = int(max(len(sgd_result.rescaled_scalars), line_count) / line_count)
                # drawer.draw_linear_regression(x, y, nth=nth, show_image=False)

    # draw_hist(list(map(lambda result: result.time)))
    # draw_hist(memory_usage)


def task_3(mod, batch_size, scaler, x, y, lr, epoch):
    return sgd(x, y, start=[0, 1], batch_size=batch_size, epoch=epoch, random_state=0,
               scheduler=ConstLRScheduler(lr),
               scaler_ctor=scaler,
               sgd_mod=mod)


def task_2(batch_size, scaler, x, y):
    return sgd(x, y, start=[0, 1], batch_size=batch_size, epoch=70, random_state=0,
               scheduler=ConstLRScheduler(0.03),
               scaler_ctor=scaler,
               sgd_mod=DefaultGradientMod())


def task_1(batch_size, scaler, x, y):
    return sgd(x, y, start=[0, 1], batch_size=batch_size, epoch=500, random_state=0,
               scheduler=ConstLRScheduler(0.0008),
               scaler_ctor=scaler,
               sgd_mod=DefaultGradientMod())


if __name__ == '__main__':
    main()
