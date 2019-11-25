# -*- coding: utf-8 -*-

"""
@author: zhangqian

@contact: 

@Created on: 2019-11-21 18:43
"""

import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import random

data_set = []
for index in range(50):
    data_set.append([random.randint(1, 50), random.randint(50, 100), 1])
for index in range(50):
    data_set.append([random.randint(50, 100), random.randint(1, 50), -1])

data_set = np.asarray(data_set)
print(data_set)
train_data = data_set[:, 0:2]
train_target = np.sign(data_set[:, 2])

test_data = [[231, -1], [4, 1], [1, -3], [321, 0]]
test_target = [1, 1, -1, 1]

plt.scatter(data_set[:, 0], data_set[:, 1], c=data_set[:, 2])
plt.show()

# 创建模型
clf = svm.SVC()
clf.fit(X=train_data, y=train_target, sample_weight=None)
result = clf.predict(test_data)
print('预测结果：', result)
