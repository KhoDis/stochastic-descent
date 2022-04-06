import numpy as np


class Scaler(object):
    def __init__(self):
        self.revert_numbers = []

    def rescale(self, scalars):
        scalars_copy = np.transpose(scalars)
        for i in range(1, len(scalars_copy)):
            scalars_copy[0] -= scalars_copy[i] * (self.revert_numbers[i - 1][0] / self.revert_numbers[i - 1][1])
        for j in range(1, len(scalars_copy)):
            scalars_copy[j] /= self.revert_numbers[j - 1][1]
        return np.transpose(scalars_copy)

    def scale(self, data):
        data_copy = np.transpose(data)
        for i in range(0, len(data_copy)):
            data_copy[i], revert = self.scale_axis(data_copy[i])
            self.revert_numbers.append(revert)
        return np.transpose(data_copy)

    @staticmethod
    def scale_axis(axis):
        axis_min = np.amin(axis)
        axis_max = np.amax(axis)
        scaled = (axis - axis_min) / (axis_max - axis_min)
        return scaled, [axis_min, axis_max - axis_min]
