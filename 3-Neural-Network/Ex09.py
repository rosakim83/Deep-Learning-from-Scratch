import sys, os
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image   # 이미지 표시에는 PIL 모듈 사용
sys.path.append(os.pardir)


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))    # 넘파이로 저장된 이미지 데이터를 PIL용 데이터 객체로 변환
    pil_img.show()


(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)   # 원래 이미지 모양으로 변형(flatten=True로 설정한 이미지는 1차원 넘파이 배열로 저장됨)
print(img.shape)

img_show(img)
