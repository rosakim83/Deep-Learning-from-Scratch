import numpy as np


def numerical_diff(f, x):   # 수치 미분
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


def function_tmp1(x0):  # x0에 대한 편미분 구하기(x0 = 3, x1 = 4)
    return x0 * x0 + 4.0 ** 2.0


def function_tmp2(x1):  # x1에 대한 편미분 구하기(x0 = 3, x1 = 4)
    return 3.0 ** 2.0 + x1 * x1


print(numerical_diff(function_tmp1, 3.0))
print(numerical_diff(function_tmp2, 4.0))
