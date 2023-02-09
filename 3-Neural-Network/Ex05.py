import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def identify_function(x):   # 항등 함수
    return x


X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)
print("1st floor input: {}".format(A1))
print("1st floor output: {}".format(Z1))
print()

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
print("2nd floor input: {}".format(A2))
print("2nd floor output: {}".format(Z2))
print()

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identify_function(A3)
print("final floor input: {}".format(A3))
print("final floor output: {}".format(Y))
