import numpy as np


class Convolution:
    def __init__(self, f_size, depth_in, depth_out):
        self.filter = [np.random.randn(f_size, f_size) for _ in range(0, depth_in) for _ in range(0, depth_out)]


