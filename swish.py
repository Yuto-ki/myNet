import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Swish:
    def __init__(self):
        self.input_s = None  # input を sigmoid に通した出力値
        self.output = None

    def forward(self, x):
        self.input_s = sigmoid(x)
        self.output = x * self.input_s
        return self.output

    def backward(self, dx):
        return (self.output + self.input_s * (1 - self.output)) * dx

    def update(self, batch_n):
        pass
