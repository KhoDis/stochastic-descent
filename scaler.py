import numpy as np


class AbstractScaler(object):
    def __init__(self, data):
        self.revert_numbers = []
        self._data = self.__scale(data)

    @property
    def data(self):
        return self._data

    def rescale(self, scalars):
        axes = np.transpose(scalars)
        for i in range(1, len(axes)):
            axes[0] -= axes[i] * (self.revert_numbers[i - 1][0] / self.revert_numbers[i - 1][1])
        for i in range(1, len(axes)):
            axes[i] /= self.revert_numbers[i - 1][1]
        return np.transpose(axes)

    def __scale(self, data):
        axes = np.transpose(data)
        for i in range(0, len(axes)):
            axes[i], revert = self._scale_axis(axes[i])
            self.revert_numbers.append(revert)
        return np.transpose(axes)

    def _scale_axis(self, axis):
        return axis, [0, 1]


class DefaultScaler(AbstractScaler):
    def __init__(self, data):
        super().__init__(data)


class MinMaxScaler(AbstractScaler):
    def __init__(self, data):
        super().__init__(data)

    def _scale_axis(self, axis):
        axis_min = np.amin(axis)
        axis_max = np.amax(axis)
        scaled = (axis - axis_min) / (axis_max - axis_min)
        return scaled, [axis_min, axis_max - axis_min]


class MeanScaler(AbstractScaler):
    def __init__(self, data):
        super().__init__(data)

    def _scale_axis(self, axis):
        axis_min = np.amin(axis)
        axis_max = np.amax(axis)
        axis_mean = np.mean(axis)
        scaled = (axis - axis_mean) / (axis_max - axis_min)
        return scaled, [axis_mean, axis_max - axis_min]


class UnitLengthScaler(AbstractScaler):
    def __init__(self, data):
        super().__init__(data)

    def _scale_axis(self, axis):
        axis_length = len(axis)
        scaled = axis / axis_length
        return scaled, [0, axis_length]
