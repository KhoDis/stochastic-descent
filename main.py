import numpy as np


def f(x, y):
    print("x", x)
    print("y", y)
    return np.sin(np.sqrt(x ** 2 + y ** 2))


def main(name):
    x = np.linspace(-6, 6, 30)
    y = np.linspace(-6, 6, 2)
    print("x", x)
    print("y", y)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)


if __name__ == '__main__':
    main("main.py")
