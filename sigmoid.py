import numpy as np


class Sigmoid:
    def __init__(self):
        self.output = None

    def make_shape(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, dx):
        return self.output * (1 - self.output) * dx

    def update(self, batch_n):
        pass
