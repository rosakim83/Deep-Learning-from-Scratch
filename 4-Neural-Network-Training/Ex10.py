import sys
import os
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient
sys.path.append(os.pardir)


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # 정규분포로 초기화. 형상이 2x3인 가중치 매개변수 하나를 인스턴스 변수로 가짐

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


net = simpleNet()
print(net.W)    # 가중치 매개변수

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)
print(np.argmax(p))     # 최댓값의 인덱스 > 정답 예측

t = np.array([0, 0, 1])     # 정답 레이블
print(net.loss(x, t))   # 손실함수 값

f = lambda w: net.loss(x, t)    # 간단한 함수는 람다 기법으로 구현
dw = numerical_gradient(f, net.W)
print(dw)   # 기울기
