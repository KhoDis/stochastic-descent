import numpy as np


class AbstractScaler(object):
    def __init__(self, data):
        self._data = data
        self.revert_numbers = []
        for _ in range(len(data[0])):
            self.revert_numbers.append([0, 1])
        print(len(self.revert_numbers))

    @property
    def data(self):
        return self._data

    def rescale(self, scalars):
        axes = np.transpose(scalars)
        for i in range(1, len(axes)):
            i_ = (self.revert_numbers[i - 1][0] / self.revert_numbers[i - 1][1])
            axes[0] -= axes[i] * i_
        for i in range(1, len(axes)):
            axes[i] /= self.revert_numbers[i - 1][1]
        return np.transpose(axes)

    def __scale(self, data):
        axes = np.transpose(data)
        for i in range(0, len(axes)):
            axes[i], revert = self.__scale_axis(axes[i])
            self.revert_numbers[i] = revert
        return np.transpose(axes)

    def __scale_axis(self, axis):
        return axis, [0, 1]


class MinMaxScaler(AbstractScaler):
    def __init__(self, data):
        super().__init__(data)

    def __scale_axis(self, axis):
        axis_min = np.amin(axis)
        axis_max = np.amax(axis)
        scaled = (axis - axis_min) / (axis_max - axis_min)
        return scaled, [axis_min, axis_max - axis_min]
