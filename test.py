import numpy as np


a = np.array([[1], [2], [3]])
b = np.array([[4], [5], [6]])

c = a * b.T
print(a)
print(b.T)
print(c)

print(np.sum(c, axis=1).reshape(-1, 1))
