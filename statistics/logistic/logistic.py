# -*- coding: utf-8 -*-

"""
@author: zhangqian

@contact: 

@Created on: 2019-11-15 17:51
"""
import numpy as np


def init_data():
    data = np.loadtxt('/Users/zhangqian/PycharmProjects/model_project/statistics/logistic/data.csv')
    x = data[:, 0:-1]
    y = data[:, -1]
    x = np.insert(x, 0, 1, axis=1)  # 特征数据集，添加1是构造常数项x0
    return x, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def update_weight(dataSet, labels):
    x = np.mat(dataSet)
    y = np.mat(labels).transpose()
    num_cycle = 500
    w = np.ones((x.shape[1], 1))
    lamba = 0.01

    for i in range(num_cycle):
        w += lamba * x.transpose() * (y - sigmoid(x * w))
    return w


x, y = init_data()

print(update_weight(x, y))
