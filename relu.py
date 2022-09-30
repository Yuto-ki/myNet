import numpy as np


class ReLU:
    def __init__(self):
        self.inputs = None

    def forward(self, x):
        self.inputs = x
        return x * (x > 0)

    def backward(self, dx):
        return np.where(self.inputs > 0, 1, 0) * dx

    def update(self, batch_n):
        pass
