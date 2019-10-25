# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D


def load_data_set(filename):
    csv_file = pd.read_csv(filename, usecols=["x1", "x2", "y"])
    # 将X,Y列表转化成矩阵
    xArr = np.array(csv_file[["x1", "x2"]].values.tolist()).astype(int)
    yArr = np.array(csv_file["y"]).astype(int)

    print("X : {0} \nY: {1} \n".format(xArr, yArr))
    return xArr, yArr


def train_perceptron_by_gram(data_mat, label_mat, eta):
    """
    训练模型 对偶模式
    :param data_mat:
    :param label_mat:
    :param eta:
    :return:
    """
    n, m = data_mat.shape
    a = [0 for i in range(n)]
    b = 0
    w = [0] * m

    gram = np.matmul(np.array(data_mat), np.array(data_mat).T)
    flag = True
    while flag:

        flag = False
        for i in range(n):
            tmp = 0
            for j in range(n):
                tmp += a[j] * label_mat[j] * gram[i, j]
            tmp += b
            if label_mat[i] * tmp <= 0:
                flag = True
                a[i] += eta
                b += label_mat[i] * eta

    for i in range(n):
        w += a[i] * data_mat[i,] * label_mat[i]
    return w, b


def train_perceptron(data_mat, label_mat, eta):
    """
        训练模型 原始模式
        eta: 步长
    """
    m, n = data_mat.shape
    w = np.zeros(n)
    b = 0

    flag = True
    while flag:
        for i in range(m):
            if np.any(label_mat[i] * (np.dot(w, data_mat[i]) + b) <= 0):
                w = w + eta * label_mat[i] * data_mat[i].T
                b = b + eta * label_mat[i]
                flag = True
                break
            else:
                flag = False

    return w, b


def plot_result(data_mat, label_mat, weight, bias):
    fig = plt.figure()

    axes1 = fig.add_subplot(121)
    axes2 = fig.add_subplot(122)

    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    for i in range(len(label_mat)):
        if (label_mat[i] == -1):
            type1_x.append(data_mat[i][0])
            type1_y.append(data_mat[i][1])

        if (label_mat[i] == 1):
            type2_x.append(data_mat[i][0])
            type2_y.append(data_mat[i][1])
    # 画点
    axes1.scatter(type1_x, type1_y, marker='x', s=20, c='red')
    axes1.scatter(type2_x, type2_y, marker='o', s=20, c='blue')  # 画点
    axes2.scatter(type1_x, type1_y, marker='x', s=20, c='red')
    axes2.scatter(type2_x, type2_y, marker='o', s=20, c='blue')

    #  (wx+b)y<0
    y = (0.1 * -weight[0] / weight[1] + -bias / weight[1], 4.0 * -weight[0] / weight[1] + -bias / weight[1])
    # 画线
    axes1.add_line(Line2D((0.1, 4), y, color="yellow"))

    axes2.add_line(Line2D((0.1, 4), y, color="blue"))

    plt.show()


if __name__ == "__main__":
    dataMat, labelMat = load_data_set("statistics/perceptron/perceptron.csv")

    weight, bias = train_perceptron(dataMat, labelMat, 0.1)

    print("原始模式：w: {0}   b: {1}".format(weight, bias))

    w, b = train_perceptron_by_gram(dataMat, labelMat, 0.1)

    print("对偶模式：w: {0}   b: {1}".format(w, b))

    plot_result(dataMat, labelMat, weight, bias)
