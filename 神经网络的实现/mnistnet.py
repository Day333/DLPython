# coding: utf-8
import pickle
import sys, os

import numpy as np

sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
from dataset.mnist import load_mnist
from PIL import Image


# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# softmax函数
def softmax(a):
    exp_a = np.exp(a)
    sum_sxp_a = np.sum(exp_a)
    y = exp_a / sum_sxp_a

    return y


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test


def init_netwok():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network_, x):
    W1, W2, W3 = network_['W1'], network_['W2'], network_['W3']
    b1, b2, b3 = network_['b1'], network_['b2'], network_['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_netwok()

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))