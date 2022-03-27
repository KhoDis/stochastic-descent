import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from function import *
from schedulers import *
from old.wolfe import line_search

mpl.use('TkAgg')

plt.rcParams["figure.figsize"] = (20, 10)


@deprecated
def convergence_epoch(scheduler, start, epoch, wolfe):
    """Gradient descent implementation with epoch"""
    points = np.zeros((epoch, get_dim() - 1))
    points[0] = start
    for i in range(1, epoch):
        point = points[i - 1]
        gradient = np.array(grad(point))

        lr = scheduler.show(i, point, -gradient)

        if wolfe:
            # draw_wolfe_conditions(point, direction, lr)
            lr = line_search(point, -gradient, lr)
            if lr is None:
                print("wolfe's conditions can't be satisfied, change beta")
                exit(1)

        points[i] = point - lr * gradient

    return points


def convergence_eps(scheduler, start, eps, wolfe):
    """Gradient descent implementation with epsilon"""
    points = [start]
    while True:
        point = points[-1]
        gradient = np.array(grad(point))

        lr = scheduler.show(len(points) - 1, point, -gradient)

        if wolfe:
            # draw_wolfe_conditions(point, direction, lr)
            lr = line_search(point, -gradient, lr)
            if lr is None:
                print("wolfe's conditions can't be satisfied, change beta")
                exit(1)

        # np.append(points, points[-1] - lr * gradient)
        points.append(points[-1] - lr * gradient)

        if abs(f(*points[-2]) - f(*points[-1])) < eps:
            break

    return points


def draw3d(points, name, index):
    """Draws 3d plot of descent"""
    last_scalar = points[-1]
    x_axis = np.linspace(last_scalar[0] - 10, last_scalar[0] + 10, 100)
    y_axis = np.linspace(last_scalar[1] - 10, last_scalar[1] + 10, 100)
    grid = np.meshgrid(x_axis, y_axis)

    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(*grid, f(*grid))

    plt.plot(points[:, 0], points[:, 1], 'o-')
    plt.contour(*grid, f(*grid), sorted([f(*p) for p in points]))
    path = f'img/3d_convergence_{name}'
    os.makedirs(path, exist_ok=True)
    # plt.title(f'Start: {start}')
    plt.savefig(f'{path}/{index}.png')
    # plt.show()
    plt.clf()


def table(name, lrs, results):
    """Table for methods stat_epoch_steps_range"""
    df = pd.DataFrame({
        "diff": results,
        "lrs": lrs
    }, index=range(1, lrs.size + 1))

    df.columns.name = name

    path = 'tables'
    os.makedirs(path, exist_ok=True)
    df.to_csv(f'{path}/{name}.csv', index=True)

    return df


def stat_epoch_steps_range(scheduler_ctor, start, epoch, steps):
    """Collecting stats for schedulers with indexed steps"""
    scheduler_name = ""
    results = []

    for index, step in enumerate(steps):
        scheduler = scheduler_ctor(step)
        scheduler_name = scheduler.name()

        points = convergence_epoch(scheduler, start, epoch, False)

        if get_dim() == 3:
            draw3d(points, scheduler_name, index)

        results.append(abs(f(*get_min_point()) - f(*points[-1])))

        # print_additional_info(scheduler_name, points)

    print(table(scheduler_name, steps, results))


def stat_epoch(scheduler_ctor, start, epoch):
    """Collecting stats for fixed amount if iterations"""
    scheduler = scheduler_ctor()
    scheduler_name = scheduler.name()

    points = convergence_epoch(scheduler, start=start, epoch=epoch, wolfe=True)

    if get_dim() == 3:
        draw3d(points, scheduler_name, '0')

    print_additional_info(scheduler_name, points)


def stat_eps(scheduler_ctor, start, eps):
    """Collecting stats for exact eps parameter (distance between two last points)"""
    scheduler = scheduler_ctor()
    scheduler_name = scheduler.name()

    points = convergence_eps(scheduler, start=start, eps=eps, wolfe=True)

    # if get_dim() == 3:
    #     draw3d(points, scheduler_name, '0')

    print_additional_info(scheduler_name, points)

    return len(points)


def print_additional_info(scheduler_name, points):
    print(f"\n{scheduler_name.upper()} scheduler")
    print(f"Current dim is {get_dim()}")
    print(f"gradient is called {get_grad_counter()} times")
    print(f"function is called {get_f_counter()} times")
    print(f"iterations: {len(points)}")
    print(f"Calculated minimum point: {points[-1]}")


# noinspection PyPep8Naming
def T(n):
    """Calculate minimum point for direction n for random function"""
    set_dim(n)
    return stat_eps(DichotomyScheduler, start=np.repeat(-100.0, n - 1), eps=1e-14)


def main():
    if get_dim() == 3:
        find_min_point()
        print(f"Actual minimum is {get_min_point()}")

    # n_dimensional_stats()

    stat_epoch_steps_range(
        ConstLRScheduler,
        start=np.repeat(-20.0, get_dim() - 1), epoch=20, steps=np.linspace(0.06, 0.1, 20))
    # stat_epoch_steps_range(
    #     lambda step: SteppedLRScheduler(0.01, step),
    #     start=np.repeat(-20.0, get_dim() - 1), epoch=20, steps=np.linspace(0.001, 0.021, 20))
    # stat_epoch_steps_range(
    #     lambda step: ExponentialLRScheduler(0.1, step),
    #     start=np.repeat(-20.0, get_dim() - 1),
    #     epoch=20,
    #     steps=np.linspace(0.00001, 0.01, 20))
    # stat_epoch(DichotomyScheduler, start=np.repeat(-20.0, get_dim() - 1), epoch=20)

    # stat_epoch_steps_range(
    #     ConstLRScheduler,
    #     start=np.repeat(-50, get_dim() - 1),
    #     epoch=20,
    #     steps=np.linspace(0.01, 0.1, 20))
    # stat_epoch(DichotomyScheduler, start=np.repeat(-20.0, get_dim() - 1), epoch=20)
    #
    # stat_eps(DichotomyScheduler, start=np.repeat(-20.0, get_dim() - 1), eps=1)


def n_dimensional_stats():
    """Plot for 6-th task"""
    data = []
    dims = range(3, 20)
    for n in dims:
        data.append(T(n))
    plt.plot(dims, data)
    plt.xlabel('n')
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
