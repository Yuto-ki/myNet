import numpy as np


class SoftPlus:
    def __init__(self):
        self.input = None

    def forward(self, x):
        self.input = x
        return np.log(1 + np.exp(x))

    def backward(self, dx):
        return 1 / (1 + np.exp(-self.input)) * dx

    def update(self, batch_n):
        pass
