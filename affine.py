import numpy as np


class Affine:
    def __init__(self, n_in, n_out, lr, init_w_type):  # init_w_type(0:Xavier, 1:He)
        if init_w_type == 0:
            self.weight = np.random.randn(n_out, n_in) / np.sqrt(n_in)
        elif init_w_type == 1:
            self.weight = np.random.randn(n_out, n_in) / np.sqrt(n_in) * np.sqrt(2)
        self.bias = np.zeros((n_out, 1))
        self.inputs = None
        self.grad_w = np.zeros(self.weight.shape)
        self.grad_b = np.zeros(self.bias.shape)
        self.lr = lr  # 学習率

    def forward(self, x):
        self.inputs = x.reshape(-1, 1)
        return np.dot(self.weight, self.inputs) + self.bias

    def backward(self, dx):
        self.grad_w += dx.reshape(-1, 1) * self.inputs.T  # (n_out, 1) * (1, n_in) = (n_out, n_in)
        self.grad_b += dx
        return np.dot(self.weight.T, dx)

    def update(self, batch_n):
        self.weight -= self.grad_w * self.lr / batch_n
        self.bias -= self.grad_b * self.lr / batch_n
        self.grad_w = np.zeros(self.weight.shape)
        self.grad_b = np.zeros(self.bias.shape)
