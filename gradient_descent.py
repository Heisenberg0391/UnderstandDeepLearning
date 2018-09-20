# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

def sigmoid_activation(x):
    # 计算给定输入的sigmoid输出
    return 1.0 / (1 + np.exp(-x))

def predict(X, W):
    # 求W点积x，可以将所有x合在一起组成X一口气计算点积
    # 点积结果每一维都是一个样本对应的得分
    # 然后对得分向量求sigmoid得到每一维对应的输出
    preds = sigmoid_activation(X.dot(W))
    print(preds)

    # 根据门限值使用阶跃函数进行二分类,得到预测类别标签向量
    preds[preds <= 0.5] = 0
    preds[preds > 0] = 1

    return preds

def plot_hyperplane(W, data, labels):
    # 先画训练集
    plt.style.use("ggplot")
    plt.figure()
    plt.title("TrainData")
    plt.scatter(data[:, 0], data[:, 1], marker="o", c=np.squeeze(labels), s=30)

    # 定义自变量和因变量
    x = np.linspace(-10, 10)
    y = (-W[0]*x - W[2]) / W[1]

    # 画图
    plt.plot(x, y, label='Hyperplane')
    plt.xlabel('x label')
    limits = [-20, 20, -20, 20]
    plt.axis(limits)
    plt.ylabel('y label')
    plt.title("Classification Result")
    plt.legend()
    plt.show()

# 训练参数
epochs = 200
alpha = 0.01

# 生成n_samples个点的N分类问题，类别数=centers
# X是样本点, y是对应类别
# 其中每个数据点都是一个n_features维的向量
# cluster_std是聚类标准差, 按照先后顺序控制每个类样本的聚合度，越大越密集
(X, y) = make_blobs(n_samples=200, n_features=2, centers=2,
                    cluster_std=[1.5, 2.5], random_state=42)

# 绘制数据集
plt.style.use("ggplot")
plt.figure()
plt.title("Origin Data")
plt.scatter(X[:, 0], X[:, 1], marker="o", c= np.squeeze(y), s= 30)
plt.legend()
plt.show()

# 从行向量变成列向量，因为样本点X是列向量
y = y.reshape((y.shape[0], 1))

# 在W最后插入一列1将W和b合并
X = np.c_[X, np.ones((X.shape[0]))]  # 在X每个数据点后增加一项表示偏置（初始化为1）

# 将数据集分为训练集和测试集
(trainX, testX, trainY, testY) = train_test_split(X, y,
                                                  test_size=0.2, random_state=1)

# 绘制测试集
plt.style.use("ggplot")
plt.figure()
plt.title("TestData")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c= np.squeeze(testY), s= 30)
plt.legend()
plt.show()

# 初始化W和损失list
print("[INFO] training...")
W = np.random.randn(X.shape[1], 1)  # 随机初始化权重（包括偏置）
losses = []

# 迭代优化
for epoch in np.arange(0, epochs):
    # 类别预测值y = sig(Wx)
    """ 这里trainX.dot(W)是把所有样本一起求权重和
        然后得到一个维度等于样本数的列向量
        每一维是某个样本的权重和
    """
    preds = sigmoid_activation(trainX.dot(W))

    # 预测误差=预测类别值-真实类别值
    error = preds - trainY
    loss = np.sum(error ** 2)  # 所有样本的平方误差之和
    losses.append(loss)

    # 先将训练集转置成3行1000列
    # 然后求梯度 = 输入特征向量点积预测误差
    gradient = trainX.T.dot(error)

    # 更新权重阶段，我们需要沿着梯度的反方向微调权重
    W += -alpha * gradient

    # 显示更新情况
    if epoch == 0 or (epoch + 1) % 50 == 0:
        print("[INFO] epoch={}, loss={:.7f}".format(int(epoch + 1),
                                                    loss))
        plot_hyperplane(W, trainX, trainY)

# 评估模型
print("[INFO] evaluating...")
# 使用训练好的模型W对测试集进行分类，得到类别标签向量
preds = predict(testX, W)
print(classification_report(testY, preds))

# 绘制损失曲线
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
