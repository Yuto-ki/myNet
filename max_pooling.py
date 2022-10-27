import numpy as np


class MaxPooling:
    def __init__(self, f_size):
        self.weight = None
        self.f_size = f_size
        self.input = None
        self.output_shape = None
        self.output = None  # 実際には不要

    def forward(self, x):
        self.input = x
        self.weight = np.zeros(x.shape)
        out = np.zeros((len(x), int(len(x[0]) / self.f_size), int(len(x[0][0]) / self.f_size)))
        for i in range(len(x)):
            for j in range(int(len(x[0]) / self.f_size)):
                for k in range(int(len(x[0][0]) / self.f_size)):
                    tmp = np.argmax(x[i, j*self.f_size:(j+1)*self.f_size, k*self.f_size:(k+1)*self.f_size])
                    self.weight[i, j*self.f_size+int(tmp/2), k*self.f_size+(tmp % 2)] = 1
                    out[:, j, k] = x[i, j*self.f_size+int(tmp/2), k*self.f_size+(tmp % 2)]
        self.output_shape = out.shape
        self.output = out  # 実際には不要
        return out

    def backward(self, dx):
        dx = dx.reshape(self.output_shape)
        dx_rs = np.zeros((len(dx), len(dx[0])*self.f_size, len(dx[0][0])*self.f_size))
        for i in range(self.f_size):
            for j in range(self.f_size):
                dx_rs[:, i::self.f_size, j::self.f_size] = dx
        return dx_rs * self.weight

    def update(self, batch_n):
        pass
