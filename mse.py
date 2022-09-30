import numpy as np


class Mse:
    @staticmethod
    def forward(neu_output, true_output):
        return np.square(neu_output - true_output.reshape(-1, 1)).mean()

    @staticmethod
    def backward(neu_output, true_output):
        return 2 * (neu_output - true_output.reshape(-1, 1)).mean()
