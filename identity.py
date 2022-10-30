class Identity:

    @staticmethod
    def make_shape(x):
        return x

    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def backward(dx):
        return dx

    def update(self, batch_n):
        pass
