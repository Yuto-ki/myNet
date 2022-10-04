import numpy as np


class SoftMax:
    def __init__(self):
        self.output = None
        self.input_size = None
        self.dh_du = None

    def forward(self, x):
        self.input_size = len(x)
        x = x - np.max(x)
        self.output = np.exp(x) / np.sum(np.exp(x))
        return self.output

    def backward(self, dx):
        self.dh_du = np.zeros((self.input_size, self.input_size))
        for i in range(0, self.input_size):
            for j in range(0, self.input_size):
                self.dh_du[i][j] = - self.output[i] * self.output[j]
        for i in range(0, self.input_size):
            self.dh_du[i][i] = self.output[i] * (1 - self.output[i])
        return np.dot(self.dh_du, dx)

    def update(self, batch_n):
        pass
