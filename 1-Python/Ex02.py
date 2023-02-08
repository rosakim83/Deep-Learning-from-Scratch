import numpy as np  # numpy를 np라는 이름으로 가져옴

x = np.array([1.0, 2.0, 3.0])   # 파이썬의 리스트를 인수로 받아 넘파이 배열 만듦
print(x)
print(type(x))
print()

y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)    # element-wise product
print(x / y)
print()

print(x / 2.0)  # broadcast
