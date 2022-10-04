import numpy as np


class CEE:
    @staticmethod
    def forward(neu_output, true_output):
        return (-1) * (true_output * np.exp(neu_output + 1e-7)).mean()

    @staticmethod
    def backward(neu_output, true_output):
        print((neu_output + 1e-7))
        return (-1) * true_output / (neu_output + 1e-7)
