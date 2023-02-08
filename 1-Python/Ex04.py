import numpy as np

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0][1])
print()

for i in X:
    print(i)
print()

X = X.flatten()  # X를 벡터로 변환(평탄화)
print(X)
print(X[np.array([0, 2, 4])])   # 인덱스를 배열로 지정
print(X > 15)   # bool 배열 생성
print(X[X > 15])
