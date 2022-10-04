# 分類問題用 ロジスティック回帰

import matplotlib.pyplot as plt
import numpy as np
import random
import math
from outputer import OutPutter
from net import Net
from cee import CEE
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import pandas as pd


def function(x):
    return 0.1 * ((x - 2) ** 2) + 5 * math.sin(x)


def oh_a(x):
    a = [0] * 3
    a[x] = 1
    return a


def oh(x):
    out_p = []
    for t in range(0, len(x)):
        out_p.append(oh_a(x[t]))
    return out_p


lr = 0.1
shape_data = [13, 20, 20, 3]
activ_f_data = [2, 2, 6]
net = Net(shape_data, activ_f_data, lr)

epoch_num = 1  # epoch数
batch_n = 10  # バッチサイズ

# 入力データ生成
data_wine = load_wine()
X = data_wine["data"]
Y = data_wine["target"]
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)
y_train = oh(y_train)
y_test = oh(y_test)
print(y_train)

n = 500  # データ数
split_rate = 5  # データ数 / テストデータ数
data_f = np.linspace(-10.0, 10.0, n)
data_res_f = list(map(function, data_f))

data_u = np.mean(data_f)  # 入力データの平均値
data_s = np.std(data_f)  # 入力データの標準偏差
data_res_u = np.mean(data_res_f)  # 入力データに対する正解の出力の平均値
data_res_s = np.std(data_res_f)  # 入力データに対する正解の出力の標準偏差
data_train = []
data_train_res = []
data_test = []
data_test_res = []
data_test_f = []
data_test_res_f = []
train_n = 0
test_n = 0

for i in range(0, n):
    if i % split_rate == 0:
        data_test_f.append(data_f[i])
        data_test_res_f.append(data_res_f[i])
        data_test.append((data_f[i] - data_u) / data_s)
        data_test_res.append((data_res_f[i] - data_res_u) / data_res_s)
        test_n += 1
    else:
        data_train.append((data_f[i] - data_u) / data_s)
        data_train_res.append((data_res_f[i] - data_res_u) / data_res_s)
        train_n += 1
data_train = np.array(data_train)
data_train_res = np.array(data_train_res)
data_test = np.array(data_test)
data_test_res = np.array(data_test_res)
data_test_f = np.array(data_test_f)
data_test_res_f = np.array(data_test_res_f)
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


for i in range(1, epoch_num+1):
    lc_mse_x.append(i)
    loss = 0
    for j in range(0, batch_n):
        k = random.randint(0, train_n-1)
        out = net.forward(X_train[k])
        loss += CEE.forward(out, y_train[k]) / batch_n
        print(CEE.backward(out, y_train[k]))
        net.backward(CEE.backward(out, y_train[k]))
    lc_mse.append(loss)
    net.update(batch_n)

    lc_x.append(i)
    loss_test = 0
    for k in range(0, test_n):
        out = net.forward(data_test[k])
        loss_test += CEE.forward(out, data_test_res[k]) / test_n
    lc_validation.append(loss_test)


plt.clf()
plt.plot(lc_mse_x, lc_mse, color='b', linewidth='1')
plt.plot(lc_x, lc_validation, color='y', linestyle="dashed", linewidth='1')
plt.savefig("per")
