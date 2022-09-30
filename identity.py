class Identity:
    @staticmethod
    def forward(x):
        return x

    @staticmethod
    def backward(dx):
        return dx

    def update(self, batch_n):
        pass
