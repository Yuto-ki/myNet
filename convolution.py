import numpy as np


class Convolution:
    def __init__(self, f_size, depth_in, depth_out, stride, h_in, w_in):
        self.lr = 0.1
        self.f_size = f_size
        self.depth_in = depth_in
        self.depth_out = depth_out
        self.height_in = h_in
        self.width_in = w_in
        self.stride = stride
        self.height_out = int((self.height_in - self.f_size) / self.stride + 1)
        self.width_out = int((self.width_in - self.f_size) / self.stride + 1)
        self.filter = np.array([[np.random.randn(f_size, f_size)
                                 for _ in range(0, depth_in)] for _ in range(0, depth_out)])
        self.bias = np.zeros((depth_out, self.height_out, self.width_out))
        self.grad_f = np.zeros(self.filter.shape)
        self.grad_b = np.zeros(self.bias.shape)
        self.input = None
        self.output = None

    def forward(self, x):
        self.input = x
        self.output = np.zeros((self.depth_out, self.height_out, self.width_out))
        for i1 in range(0, self.depth_out):
            for i2 in range(0, self.height_out):
                for i3 in range(0, self.width_out):
                    for i4 in range(0, self.depth_in):
                        for i5 in range(0, self.f_size):
                            for i6 in range(0, self.f_size):
                                self.output[i1][i2][i3] += x[i4][i2 * self.stride + i5][i3 * self.stride + i6] * \
                                                           self.filter[i1][i4][i5][i6]
        self.output += self.bias
        return self.output

    def backward(self, dx):
        grad_out = np.zeros(self.input.shape)
        self.grad_b = dx
        a = self.height_in - self.f_size + 1
        b = self.width_in - self.f_size + 1
        for i in range(0, self.f_size):
            for j in range(0, self.f_size):
                print((self.input[:, i:a + i:self.stride, j:b + j:self.stride] * dx).shape)
                self.grad_f[:, :, i, j] += np.sum(np.sum(self.input[:, i:a + i:self.stride, j:b + j:self.stride] *
                                                         dx, axis=2), axis=1)
        a = self.f_size * self.stride
        for i in range(0, self.height_out):
            for j in range(0, self.width_out):
                print(self.filter.shape)
                print(dx[:, i:i + self.f_size, j:j + self.f_size].shape)
                print(self.filter * dx[:, i:i + self.f_size, j:j + self.f_size])
                print(grad_out[:, i:a + i:self.stride, j:a + j:self.stride].shape)
                grad_out[:, i:a + i:self.stride, j:a + j:self.stride] += \
                    self.filter * dx[:, i:i + self.f_size, j:j + self.f_size]
        return grad_out

    def update(self, batch_n):
        self.filter -= self.grad_f * self.lr / batch_n
        self.bias -= self.grad_b * self.lr / batch_n
        self.grad_f = np.zeros(self.filter.shape)
        self.grad_b = np.zeros(self.bias.shape)
