import sys, os
import numpy as np
from dataset.mnist import load_mnist
sys.path.append(os.pardir)


# 정답 레이블이 one-hot 인코딩일 경우 교차 엔트로피 오차 구현
def cross_entropy_error_1(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(t * np.log(y + 1e-7)) / batch_size


# 정답 레이블이 숫자 레이블일 경우 교차 엔트로피 오차 구현
def cross_entropy_error_2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size    # 정답 레이블에 해당하는 y 출력값 추출됨


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print(x_train.shape)
print(t_train.shape)

# mini-batch 학습 구현
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)   # mini-batch 데이터의 인덱스 뽑아냄
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]
