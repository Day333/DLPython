# 项目文档


## 说明

此项目包含了一个用于识别手写数字的神经网络模型。以下是代码中各部分的详细说明：

### 1. 导入库和模块

代码以导入必要的库和模块开始，确保代码能够正常运行。`sys.path.append(os.pardir)`语句用于添加父目录到系统路径，以便导入父目录的文件。

### 2. 定义激活函数

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```
这是sigmoid激活函数的定义，用于将神经网络的激活值映射到0到1之间。

### 3. 定义softmax函数
```python
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
```
这是softmax激活函数的定义，用于将神经网络的输出转化为概率分布。

### 4. 定义图像显示函数
```python
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
```
这个函数用于显示图像，它将NumPy数组表示的图像转换为PIL图像并显示出来。

### 5. 加载测试数据
```python
def get_data():
    """
    从MNIST数据集加载测试数据
    返回:
        x_test: 测试图像数据
        t_test: 测试标签数据
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    return x_test, t_test
```
这个函数从MNIST数据集中加载测试数据，返回图像数据和相应的标签数据。

### 6. 初始化神经网络模型
```python
def init_network():
    """
    初始化神经网络模型，加载预训练的权重数据
    返回:
        network: 包含模型权重的字典
    """
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network
```
这个函数初始化神经网络模型，它加载了预训练的权重数据并返回一个包含模型权重的字典。

### 7. 进行数字识别
```python
def predict(network_, x):
    """
    使用神经网络模型进行数字识别
    参数:
        network_: 包含模型权重的字典
        x: 输入图像数据
    返回:
        y: 预测结果概率分布
    """
    W1, W2, W3 = network_['W1'], network_['W2'], network_['W3']
    b1, b2, b3 = network_['b1'], network_['b2'], network_['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y
```
这个函数用于进行数字识别，它接受神经网络模型和输入图像数据作为参数，返回预测结果的概率分布。

### 8. 加载数据和进行数字识别
```python
x, t = get_data()
network = init_network()

batch_size = 100 # 批数量
accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i + batch_size])

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
```
这个部分加载数据、初始化网络模型，然后使用批处理的方式进行数字识别，最后计算并打印准确率。