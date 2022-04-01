import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt, cm

from dataset_reader import DatasetReader
from old.search_algorithms import Dichotomy

mpl.use('TkAgg')


def f(point, x, y):
    # print('f_point', point)
    # print('f_x', x)
    # print('f_y', y)
    accumulator = 0
    for j in range(0, len(y)):
        prediction = point[0]
        for i in range(1, len(point)):
            prediction += x[j][i - 1] * point[i]
        accumulator += (y[j] - prediction) ** 2
    return accumulator


def draw_2d_surface(points, x, y):
    shift = 5

    # shift = 2
    # last_point = points[-1]
    x_axis = np.linspace(-5, 5, 1000)
    y_axis = np.linspace(-5, 5, 1000)
    # x_axis = np.linspace(last_point[0] - shift, last_point[0] + shift, 1000)
    # y_axis = np.linspace(last_point[1] - shift, last_point[1] + shift, 1000)
    zp = np.ndarray((1000, 1000))
    # grid = np.meshgrid(x_axis, y_axis)
    # ax = plt.figure().add_subplot(projection='3d')
    # ax.plot_surface(*grid, f(grid, x, y))

    # TODO: try out axis swapping
    # TODO: annotate contours
    for i in range(0, len(x_axis)):
        for j in range(0, len(y_axis)):
            zp[j][i] = f([x_axis[i], y_axis[j]], x, y)

    plt.plot(points[:, 0], points[:, 1], 'o-')
    # for i, point in enumerate(points):
    #     plt.text(*point, f'{i}')
    # print('x', x)
    # print(f'{x=}')
    # print('y', y)
    # print('points', points)
    # print('grid', np.array(grid))
    # print('z', z)
    levels = sorted([f(p, x, y) for p in points])
    plt.contour(x_grid, y_grid, f([x_grid, y_grid], x, y), levels=levels)
    plt.show()
    plt.clf()


def grad(x_batch, y_batch, point):
    h = 1e-5
    result = np.zeros(point.size)
    for i, n in enumerate(point):
        point[i] = point[i] + h
        f_plus = f(point, x_batch, y_batch)
        point[i] = point[i] - 2 * h
        f_minus = f(point, x_batch, y_batch)
        point[i] = point[i] + h
        result[i] = (f_plus - f_minus) / (2 * h)
    return result


def scale(axis):
    axis_min = np.amin(axis)
    axis_max = np.amax(axis)
    scaled = (axis - axis_min) / (axis_max - axis_min)
    return scaled, (lambda vector: vector * (axis_max - axis_min) + axis_min)


def sgd(
        gradient, x, y, start, batch_size=1, epoch=50,
        tolerance=1e-06, random_state=None, dtype="float64"):
    dtype_ = np.dtype(dtype)

    x = np.array(x, dtype=dtype_)
    y = np.array(y, dtype=dtype_)

    n_obs = len(x)

    # x_reverts = []
    # x_axes = []
    # for x_row in np.transpose(x):
    #     x_row_scaled, revert = scale(x_row)
    #     x_axes.append(x_row_scaled)
    #     x_reverts.append(revert)
    #
    # y, y_revert = scale(y)
    # x = np.array(np.transpose(x_axes), dtype=dtype_)

    # Initializing the random number generator for shuffling (link further)
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    # Initializing the values of the variables
    current_point = np.array(start, dtype=dtype_)

    # Setting up and checking the learning rate
    # learn_rate = np.array(learn_rate, dtype=dtype_)
    # if np.any(learn_rate <= 0):
    #     raise ValueError("'learn_rate' must be greater than zero")

    # Setting up and checking the size of minibatches
    # Making sure they are not float
    batch_size = int(batch_size)

    # batch_size = {1} => SGD
    # batch_size = [2, n_obs) => minibatch
    # batch_size = {n_obs} => GD
    if not 0 < batch_size <= n_obs:
        raise ValueError("'batch_size' must be greater than zero and less than or equal to the number of observations")

    # Setting up and checking the maximal number of iterations
    epoch = int(epoch)
    if epoch <= 0:
        raise ValueError("'epoch' must be greater than zero")

    # Setting up and checking the tolerance
    # We need it to check if the absolute difference is small enough
    # We want to ignore absurdly different points to not to break our predictions
    tolerance = np.array(tolerance, dtype=dtype_)
    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    start = np.array(start, dtype=dtype_)

    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    # Performing the gradient descent loop
    elements = [start]
    for i in range(epoch):
        # print(f'epoch={i}')
        # Shuffle x and y
        # https://www.quora.com/Why-do-we-need-to-shuffle-inputs-for-stochastic-gradient-descent
        # Shuffles n-dim dataset without breaking points
        rng.shuffle(xy)

        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):
            # print(f'start={start}')
            # Calculates interval [    start|-------|stop    ] of the batch
            stop = start + batch_size

            # TODO: We need to expand this to n dimensions
            # n_batch = get_batch(dataset, start, stop)
            # x_batch, y_batch = dataset[start:stop, :-1], dataset[start:stop, -1:]

            # Recalculating the difference
            # TODO: We need to pass to the gradient n dimensional batch and current_point (2 args). Code is below
            # grad = np.array(gradient(n_batch, current_point), dtype_)
            grad_ = np.array(gradient(xy[start:stop, :-1], xy[start:stop, -1:], current_point), dtype_)

            # print('current_point', current_point)
            # print('-grad_', -grad_)
            diff = -Dichotomy(current_point, -grad_, xy[start:stop, :-1], xy[start:stop, -1:]).calculate() * grad_

            # # Checking if the absolute difference is small enough
            # if np.all(np.abs(diff) <= tolerance):
            #     break

            # Updating the values of the variables
            current_point += diff
            elements.append(current_point.copy())

    # draw_2d(elements, x, y)

    # draw_2d_surface(np.array(elements), x, y)
    return elements  # TODO: rename


def main():
    x, y = DatasetReader('planar').data
    # x_batch = [[1, 2, 3], [1, 2, 3], ..., ] # n-1 мерная точка
    # y_batch = [1, 2, ..., ]
    scalars = sgd(grad, x, y, start=[0, 1], batch_size=5, epoch=50, random_state=0)
    # print(np.array(scalars))
    draw_2d_surface(np.array(scalars), x, y)


def draw_2d(scalars, x, y):
    plt.title('Gradient Descent')
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    for point_index in range(0, len(x)):
        plt.scatter(x[point_index][0], y[point_index], c="green")
    color = iter(cm.rainbow(np.linspace(0, 1, len(scalars))))
    for scalar in scalars:
        c = next(color)
        x = np.linspace(0, 10, 10)
        y = scalar[0]
        for i in range(1, len(scalar)):
            y += x * scalar[i]
        plt.plot(x, y, color=c)
    plt.show()


if __name__ == '__main__':
    main()
