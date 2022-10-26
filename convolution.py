import numpy as np


class Convolution:
    def __init__(self, f_size, depth_out, stride):
        self.lr = 0.1  # 初期化出来るように要変更
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

    def forward(self, x):
        # 入力依存のメンバ変数を指定
        self.depth_in = len(x)
        self.filter = np.array([[np.random.randn(self.f_size, self.f_size) for _ in range(0, self.depth_in)]
                                for _ in range(0, self.depth_out)])
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
        dx = dx.reshape(self.output.shape)
        self.grad_b = dx
        a = self.height_in - self.f_size + 1
        b = self.width_in - self.f_size + 1
        for d_o in range(0, self.depth_out):
            for d_i in range(0, self.depth_in):
                for i in range(0, self.f_size):
                    for j in range(0, self.f_size):
                        self.grad_f[d_o, d_i, i, j] += \
                            np.sum(self.input[d_i, i:a + i:self.stride, j:b + j:self.stride] * dx[d_o])
        a = self.f_size * self.stride
        for d_o in range(0, self.depth_out):
            for i in range(0, self.height_out):
                for j in range(0, self.width_out):
                    grad_out[:, i:a + i:self.stride, j:a + j:self.stride] += \
                        self.filter[d_o] * dx[d_o, i:i + self.f_size, j:j + self.f_size]
        return grad_out

    def update(self, batch_n):
        self.filter -= self.grad_f * self.lr / batch_n
        self.bias -= self.grad_b * self.lr / batch_n
        self.grad_f = np.zeros(self.filter.shape)
        self.grad_b = np.zeros(self.bias.shape)
