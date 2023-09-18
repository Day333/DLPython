import numpy as np


# 激活函数
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# 恒等函数，对应激活函数
def identity_function(a3):
    return a3


# softmax函数
def softmax(a):
    exp_a = np.exp(a)
    sum_sxp_a = np.sum(exp_a)
    y = exp_a / sum_sxp_a

    return y


# 生成网络权重
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network


# 前向传播
def forward(network_, x):
    W1, W2, W3 = network_['W1'], network_['W2'], network_['W3']
    b1, b2, b3 = network_['b1'], network_['b2'], network_['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)