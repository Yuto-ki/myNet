import numpy as np


class TanH:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
        return self.output

    def backward(self, dx):
        return (1 - self.output * self.output) * dx

    def update(self, batch_n):
        pass
