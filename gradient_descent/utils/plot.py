import matplotlib.pyplot as plt
import numpy as np


def plot_hyperplane(W, data, labels, title):
    # 先画训练集
    plt.style.use("ggplot")
    plt.figure()
    plt.title("TrainData")
    plt.scatter(data[:, 0], data[:, 1], marker="o", c=np.squeeze(labels), s=30)

    # 定义自变量和因变量
    x = np.linspace(-10, 10)
    y = (-W[0]*x - W[2]) / W[1]

    # 画图
    plt.plot(x, y, label='Hypterplane')
    plt.xlabel('x label')
    limits = [-2, 2, -2, 2]
    plt.axis(limits)
    plt.ylabel('y label')
    plt.title(title)
    plt.legend()
    plt.show()