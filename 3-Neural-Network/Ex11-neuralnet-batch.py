import sys, os
import numpy as np
import pickle
from dataset.mnist import load_mnist
sys.path.append(os.pardir)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]     # 0~99, 100~199, 200~299 ... 9900~9999
    y_batch = predict(network, x_batch)     # 100 x 10 matrix
    p = np.argmax(y_batch, axis=1)  # 100(0dim) x 10(1dim) matrix 중 1번째 차원(10)을 구성하는 각 원소에서 최댓값의 인덱스 구함
    accuracy_cnt += np.sum(p == t[i:i+batch_size])  # p.shape = (100,). 넘파이 배열끼리 비교 후 bool 배열 생성 및 true의 개수 구함

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
