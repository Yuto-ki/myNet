import numpy as np


class Convolution:
    def __init__(self, f_size, depth_out, stride):
        self.lr = 0.1
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
        # 入力依存のメンバ変数を指定 __init__に移行する必要あり
        self.depth_in = len(x)
        self.filter = np.array([[np.random.randn(self.f_size, self.f_size) for _ in range(0, self.depth_in)]
                                for _ in range(0, self.depth_out)]) / (self.f_size**2 * self.depth_in)
        self.grad_f = np.zeros(self.filter.shape)
        self.height_in = len(x[0])
        self.width_in = len(x[0][0])
        self.height_out = int((self.height_in - self.f_size) / self.stride + 1)
        self.width_out = int((self.width_in - self.f_size) / self.stride + 1)
        self.bias = np.zeros((self.depth_out, self.height_out, self.width_out))
        self.grad_b = np.zeros(self.bias.shape)

        # forwardのmain
        self.input = x
        self.output = np.zeros((self.depth_out, self.height_out, self.width_out))
        return self.output

    def forward(self, x):
        self.input = x
        self.output = np.zeros((self.depth_out, self.height_out, self.width_out))
        for i2 in range(self.height_out):
            for i3 in range(self.width_out):
                self.output[:, i2, i3] += np.sum(np.sum(np.sum(x[:, i2*self.stride:i2*self.stride+self.f_size, i3*self.stride:i3*self.stride+self.f_size] * self.filter, axis=3), axis=2), axis=1)
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
        dx = dx.reshape(self.output.shape)
        self.grad_b += dx
        for d_o in range(0, self.depth_out):
            for i in range(0, self.height_out):
                for j in range(0, self.width_out):
                    grad_out[:, i:a + i:self.stride, j:a + j:self.stride] += \
                        self.filter[d_o] * dx[d_o, i, j]
        return grad_out

    def update(self, batch_n):
        self.filter -= self.grad_f * self.lr / batch_n
        self.bias -= self.grad_b * self.lr / batch_n
        self.grad_f = np.zeros(self.filter.shape)
        self.grad_b = np.zeros(self.bias.shape)
