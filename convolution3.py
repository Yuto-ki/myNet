import numpy as np


class Convolution:
    def __init__(self, f_size, depth_out, stride, lr):
        self.lr = lr
        self.f_size = f_size
        self.depth_in = 0
        self.depth_out = depth_out
        self.height_in = None
        self.width_in = None
        self.stride = stride
        self.height_out = 0
        self.width_out = 0
        self.filter = None
        self.bias = None
        self.grad_f = None
        self.grad_b = None
        self.input = None
        self.output = None

    def make_shape(self, x):
        self.depth_in = len(x)
        self.filter = np.array([[np.random.randn(self.f_size, self.f_size) for _ in range(0, self.depth_in)]
                                for _ in range(0, self.depth_out)]) / self.f_size**2
        self.grad_f = np.zeros(self.filter.shape)
        self.height_in = len(x[0])
        self.width_in = len(x[0][0])
        self.height_out = int((self.height_in - self.f_size) / self.stride + 1)
        self.width_out = int((self.width_in - self.f_size) / self.stride + 1)
        self.bias = np.zeros((self.depth_out, self.height_out, self.width_out))
        self.grad_b = np.zeros(self.bias.shape)

        self.input = x
        self.output = np.zeros((self.depth_out, self.height_out, self.width_out))
        return self.output

    def forward(self, x):
        self.input = x
        self.output = np.zeros((self.depth_out, self.height_out, self.width_out))
        for i in range(0, self.height_out):
            for j in range(0, self.width_out):
                self.output[:, i, j] = np.sum(np.sum(np.sum(x[:, i * self.stride:i * self.stride + self.f_size, j * self.stride:j * self.stride + self.f_size] * self.filter, axis=3), axis=2), axis=1)
        self.output += self.bias
        return self.output

    def backward(self, dx):
        grad_out = np.zeros(self.input.shape)
        dx = dx.reshape((self.depth_out, 1, self.height_out, self.width_out))
        a = self.height_in - self.f_size + 1
        b = self.width_in - self.f_size + 1
        for i in range(0, self.f_size):
            for j in range(0, self.f_size):
                self.grad_f[:, :, i, j] += \
                    np.sum(np.sum(self.input[:, i:a+i:self.stride, j:b+j:self.stride] * dx, axis=3), axis=2)
        a = self.f_size * self.stride
        for i in range(self.height_out):
            for j in range(self.width_out):
                grad_out[:, i:a + i:self.stride, j:a + j:self.stride] += \
                    np.sum((self.filter.reshape((self.depth_out, -1)) * dx[:, :, i, j]).reshape((self.depth_out, self.depth_in, self.f_size, self.f_size)), axis=0)
        dx = dx.reshape(self.output.shape)
        self.grad_b += dx
        return grad_out

    def update(self, batch_n):
        self.filter -= self.grad_f * self.lr / batch_n
        self.bias -= self.grad_b * self.lr / batch_n
        self.grad_f = np.zeros(self.filter.shape)
        self.grad_b = np.zeros(self.bias.shape)
