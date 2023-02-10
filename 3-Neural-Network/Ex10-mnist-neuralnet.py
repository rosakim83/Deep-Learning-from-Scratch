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
accuracy_cnt = 0
for i in range(len(x)):  # 0 ~ 10000
    y = predict(network, x[i])  # 숫자 0 ~ 9까지의 확률을 담은 넘파이 배열
    p = np.argmax(y)    # 확률이 가장 높은 원소의 인덱스를 얻음 = 신경망 예측 결과
    if p == t[i]:   # 신경망이 예측한 답변과 정답 레이블 비교
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
