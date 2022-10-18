import numpy as np


class Convolution:
    def __init__(self, f_size, depth_in, depth_out, stride, h_in, w_in):
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
        self.out = None

    def forward(self, x):
        self.out = np.zeros((self.depth_out, self.height_out, self.width_out))
        for i1 in range(0, self.depth_out):
            for i2 in range(0, self.height_out):
                for i3 in range(0, self.width_out):
                    for i4 in range(0, self.depth_in):
                        for i5 in range(0, self.f_size):
                            for i6 in range(0, self.f_size):
                                self.out[i1][i2][i3] += x[i4][i2 * self.stride + i5][i3 * self.stride + i6] * \
                                                        self.filter[i1][i4][i5][i6]
        self.out += self.bias
        return self.out
