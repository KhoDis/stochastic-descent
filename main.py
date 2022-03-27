import numpy as np

from function import Function


def ssr_gradient(x, y, b):
    res = b[0] + b[1] * x - y
    return res.mean(), (res * x).mean()  # .mean() is a method of np.ndarray


def sgd(
        gradient, x, y, start, learn_rate=0.1, batch_size=1, n_iter=50,
        tolerance=1e-06, dtype="float64", random_state=None
):
    # Setting up the data type for NumPy arrays
    dtype_ = np.dtype(dtype)

    # Converting x and y to NumPy arrays
    x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
    n_obs = x.shape[0]
    if n_obs != y.shape[0]:
        raise ValueError("'x' and 'y' lengths do not match")
    xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

    # Initializing the random number generator
    seed = None if random_state is None else int(random_state)
    rng = np.random.default_rng(seed=seed)

    # Initializing the values of the variables
    vector = np.array(start, dtype=dtype_)

    # Setting up and checking the learning rate
    learn_rate = np.array(learn_rate, dtype=dtype_)
    if np.any(learn_rate <= 0):
        raise ValueError("'learn_rate' must be greater than zero")

    # Setting up and checking the size of minibatches
    batch_size = int(batch_size)

    if not 0 < batch_size <= n_obs:
        raise ValueError("'batch_size' must be greater than zero and less than or equal to the number of observations")

    # Setting up and checking the maximal number of iterations
    n_iter = int(n_iter)
    if n_iter <= 0:
        raise ValueError("'n_iter' must be greater than zero")

    # Setting up and checking the tolerance
    tolerance = np.array(tolerance, dtype=dtype_)

    if np.any(tolerance <= 0):
        raise ValueError("'tolerance' must be greater than zero")

    # Performing the gradient descent loop
    for _ in range(n_iter):
        # Shuffle x and y
        rng.shuffle(xy)

        # Performing minibatch moves
        for start in range(0, n_obs, batch_size):

            stop = start + batch_size
            x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

            # Recalculating the difference
            grad = np.array(gradient(x_batch, y_batch, vector), dtype_)

            diff = -learn_rate * grad

            # Checking if the absolute difference is small enough
            if np.all(np.abs(diff) <= tolerance):
                break

            # Updating the values of the variables
            vector += diff

    return vector if vector.shape else vector.item()


def predict(row, coefficients):
    prediction = coefficients[0]
    for i in range(len(row) - 1):
        prediction += coefficients[i + 1] * row[i]
    return prediction


def coefficients_sgd(f: Function, start, learn_rate=0.1, epoch=20):
    coefficients = np.repeat(0.0, len(data_points))
    for _ in range(1, epoch):
        for point in data_points:
            last_point = point

            gradient = np.array(f.grad(last_point))

            error_prediction = predict(last_point, coefficients) - last_point[-1]  # pig?

            coefficients[0] = coefficients[0] - learn_rate * error_prediction
            for i in range(len(last_point) - 1):
                coefficients[i + 1] = coefficients[i + 1] - learn_rate * error_prediction * last_point[i]

        # # Add new point to descent
        # points.append(last_point - learn_rate * error_prediction * gradient)

    return points


def linear_regression_sgd(train, test, l_rate, n_epoch):
    predictions = list()
    coefficients = coefficients_sgd(train, l_rate, n_epoch)
    for row in test:
        yhat = predict(row, coefficients)
        predictions.append(yhat)
    return predictions


def main(name):
    # sgd(ssr_gradient, np.linspace(0, 1, 100), np.linspace(0, 2, 100), np.repeat(0, 100))

    # cubic_function = Function(lambda x, y: 7 * (x - 1) ** 2 + 3 * (y + 8) ** 2 + 3 * x * y, 3)
    # pig_sgd(cubic_function)



if __name__ == '__main__':
    main("piggyback.py")
