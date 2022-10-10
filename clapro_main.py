# 分類問題用 ロジスティック回帰

import matplotlib.pyplot as plt
import numpy as np
import random
import math
from outputer import OutPutter
from net import Net
from sce import SCE
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def function(x):
    return 0.1 * ((x - 2) ** 2) + 5 * math.sin(x)


def oh_a(x):
    a = [0] * 3
    a[x] = 1
    return np.array([a]).T


def oh(x):
    out_p = []
    for t in range(0, len(x)):
        out_p.append(oh_a(x[t]))
    return out_p


# ベクトルの最大要素のインデックスを返す
def oh_t_sc(x):
    return np.argmax(x)


lr = 0.1
shape_data = [13, 3, 3]
activ_f_data = [1, 0]
net = Net(shape_data, activ_f_data, lr)

epoch_num = 1000  # epoch数
batch_n = 3  # バッチサイズ

# 入力データ生成
data_wine = load_wine()
X = data_wine["data"]
# X = np.delete(X, 0, axis=1)
# X = np.delete(X, 0, axis=1)
# X = np.delete(X, 0, axis=1)
# X = np.delete(X, 0, axis=1)
# X = np.delete(X, 0, axis=1)
# X = np.delete(X, 0, axis=1)
# X = np.delete(X, 0, axis=1)
# X = np.delete(X, 0, axis=1)
# X = np.delete(X, 0, axis=1)
# X = np.delete(X, 1, axis=1)
# X = np.delete(X, 1, axis=1)
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
Y = data_wine["target"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
y_train = oh(y_train)
y_test = oh(y_test)


# X_train = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# y_train = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
# X_test = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
# y_test = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# 入力データ生成終了

train_n = len(X_train)
test_n = len(X_test)

show_per_n = 2
test_deter = show_per_n
graph_n = math.floor(math.log2(epoch_num)) + 1
oup = OutPutter(graph_n, 'result (n : number of train)')
fig_count = 0

lc_mse_x = []
lc_mse = []
lc_x = []
lc_validation = []
accuracy = []

sce = SCE()

for i in range(1, epoch_num+1):
    lc_mse_x.append(i)
    loss = 0
    for j in range(0, batch_n):
        k = random.randint(0, train_n-1)
        out = sce.cal_out(net.forward(X_train[k]))
        loss += sce.cal_loss(y_train[k]) / batch_n
        net.backward(sce.backward(y_train[k]))
    lc_mse.append(loss)
    net.update(batch_n)

    lc_x.append(i)
    loss_test = 0
    score = 0
    for k in range(0, test_n):
        out = sce.cal_out(net.forward(X_test[k]))
        loss_test += sce.cal_loss(y_test[k]) / test_n
        if oh_t_sc(out) == oh_t_sc(y_test[k]):
            score += 1
    accuracy.append(score / test_n)
    lc_validation.append(loss_test)


for k in range(0, test_n):
    out = sce.cal_out(net.forward(X_test[k]))
    print(out)
    print(y_test[k])

plt.clf()
plt.plot(lc_x, accuracy, color='y', linewidth='1')
plt.savefig("per")
