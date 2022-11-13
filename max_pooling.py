import numpy as np


class MaxPooling:
    def __init__(self, f_size, stride):
        self.f_size = f_size
        self.stride = stride
        self.input = None
        self.depth_out = 0
        self.height_out = 0
        self.width_out = 0
        self.output_shape = None
        self.where = None

    def make_shape(self, x):
        self.input = x
        self.depth_out = len(x)
        self.height_out = int((len(x[0]) - self.f_size) / self.stride + 1)
        self.width_out = int((len(x[0][0]) - self.stride) / self.stride + 1)
        out = np.zeros((self.depth_out, self.height_out, self.width_out))
        self.where = np.zeros((self.depth_out, self.height_out, self.width_out))
        self.output_shape = out.shape
        return out

    def forward(self, x):
        self.input = x
        self.where = np.zeros((self.depth_out, self.height_out, self.width_out))
        out = np.zeros((len(x), int(len(x[0]) / self.f_size), int(len(x[0][0]) / self.f_size)))
        for i in range(self.depth_out):
            for j in range(self.height_out):
                for k in range(self.width_out):
                    tmp = np.argmax(x[i, j*self.stride:j*self.stride+self.f_size, k*self.stride:k*self.stride+self.f_size])
                    self.where[i, j, k] = [i, ]
                    out[:, j, k] = x[i, j*self.f_size+int(tmp/self.f_size), k*self.f_size+(tmp % self.f_size)]
        self.output_shape = out.shape
        return out
        # self.input = x
        # out = np.zeros((len(x), int((len(x[0])-self.f_size+1)/self.stride), int((len(x[0][0])-self.f_size+1)/self.stride)))
        # self.where = np.zeros(out.shape)
        # for i in range(len(x)):
        #     for j in range(int((len(x[0])-self.f_size+1)/self.stride)):
        #         for k in range(int((len(x[0][0])-self.f_size+1)/self.stride)):
        #             tmp = np.argmaxx[i, j*self.stride:j*self.stride+self.f_size-1, k*self.stride:k*self.stride+self.f_size-1]
        #             self.where[i, j, k] = []



    def backward(self, dx):
        dx = dx.reshape(self.output_shape)
        dx_rs = np.zeros((len(dx), len(dx[0])*self.f_size, len(dx[0][0])*self.f_size))
        for i in range(self.f_size):
            for j in range(self.f_size):
                dx_rs[:, i::self.f_size, j::self.f_size] = dx
        return dx_rs * self.weight

    def update(self, batch_n):
        pass
