# Softmax Cross Entropy
from softmax import SoftMax
from cee import CEE


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
        print(self.out - t)
        return self.out - t
