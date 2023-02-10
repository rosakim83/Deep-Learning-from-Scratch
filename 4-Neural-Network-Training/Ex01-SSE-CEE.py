import numpy as np


def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


def cross_entropy_error(y, t):
    delta = 1e-7    # np.log 함수에 0이 들어가지 않도록 아주 작은 값 더함
    return -np.sum(t * np.log(y + delta))


t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y1 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
y2 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(sum_squares_error(np.array(y1), np.array(t)))     # 첫 번째 추정 결과가 오차가 더 작으니 정답일 가능성 높음
print(sum_squares_error(np.array(y2), np.array(t)))
print(cross_entropy_error(np.array(y1), np.array(t)))   # 첫 번째 추정 결과가 오차가 더 작으니 정답일 가능성 높음
print(cross_entropy_error(np.array(y2), np.array(t)))
