# Softmax Cross Entropy
from softmax import SoftMax
from cee import CEE
import numpy as np


sfm = SoftMax()
cee = CEE()


class SCE:
    def __init__(self):
        self.out = None

    def cal_out(self, x):
        self.out = sfm.forward(x)
        return self.out

    def cal_loss(self, t):
        return cee.forward(self.out, t)

    def backward(self, t):
        tmp = self.out.T * t
        for i in range(0, len(t)):
            tmp[i][i] = t[i] * (self.out[i] - 1)
        return np.sum(tmp, axis=1).reshape(-1, 1)
