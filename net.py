from affine import Affine
from sigmoid import Sigmoid
from relu import ReLU
from tanh import TanH
from swish import Swish
from softplus import SoftPlus
from identity import Identity
from softmax import SoftMax
from convolution import Convolution
from max_pooling import MaxPooling


def init_w_type(x):
    if x == 0:
        return 0
    elif x == 1:
        return 0
    elif x == 2:
        return 1
    elif x == 3:
        return 0
    elif x == 4:
        return 1
    elif x == 5:
        return 1
    elif x == 6:
        return 0


class Net:
    def __init__(self, co_data, shape_data, activ_f_type, lr):
        # co_data: 畳み込み層の情報((1のclass指定, 必要な引数1, 必要な引数2, ..), (2のclass指定, 必要な引数1, ..), (3..))
        # class指定: 0:Convolution 1:MaxPooling 2:ReLU
        # active_f_type: 活性化関数を指定(0: identity, 1:sigmoid, 2: ReLU, 3: tanH, 4: swish, 5: softPlus, 6: softmax)
        # init_w_type: 重みの初期化方法を指定(0:Xavier, 1:He)
        self.layer = []
        self.lr = lr
        self.out = None

        # Convolutionとpooling層, 活性化層をlayerに追加
        for i in co_data:
            if i[0] == 0:
                self.layer.append(Convolution(i[1], i[2], i[3]))
            elif i[0] == 1:
                self.layer.append(MaxPooling(i[1]))
            elif i[0] == 2:
                self.layer.append(ReLU())

        # Affineと活性化層をlayerに追加
        for i in range(0, len(shape_data)-1):
            self.layer.append(Affine(shape_data[i], shape_data[i+1], self.lr, init_w_type(activ_f_type[i])))
            if activ_f_type[i] == 0:
                self.layer.append(Identity())
            elif activ_f_type[i] == 1:
                self.layer.append(Sigmoid())
            elif activ_f_type[i] == 2:
                self.layer.append(ReLU())
            elif activ_f_type[i] == 3:
                self.layer.append(TanH())
            elif activ_f_type[i] == 4:
                self.layer.append(Swish())
            elif activ_f_type[i] == 5:
                self.layer.append(SoftPlus())
            elif activ_f_type[i] == 6:
                self.layer.append(SoftMax())

    def forward(self, inputs):
        a = inputs
        for i in self.layer:
            a = i.forward(a)
        self.out = a
        return self.out

    def backward(self, loss_f_der):
        dx = loss_f_der
        for i in self.layer[::-1]:
            dx = i.backward(dx)

    def update(self, batch_n):
        for i in self.layer[::-1]:
            i.update(batch_n)
