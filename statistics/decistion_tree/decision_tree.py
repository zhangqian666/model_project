# encoding = utf-8
import operator

import pandas as pd
import numpy as np
from pandas import DataFrame


##### 使用CART 算法 ###########

class GiniData:
    def __init__(self):
        self.type = None
        self.type_value = None
        self.gini_rate = None
        self.col_index = None

    def __str__(self):
        return "type:" + self.type + "\ntype_value:" + str(
            self.type_value) + "\ngini_rate:" + str(self.gini_rate) + "\ncol_index:" + str(self.col_index)


class TNode:
    def __init__(self):
        self.type = None
        self.type_value = None
        self.left = None
        self.left_list = None
        self.right_list = None
        self.right = None

    def __str__(self):
        str1 = "type:" + str(self.type) + "; type_value:" + str(self.type_value)

        if self.left_list is not None:
            str1 += "; left_list:" + str(len(self.left_list))
        if self.right_list is not None:
            str1 += "; right_list:" + str(len(self.right_list))

        return str1


class Tree:
    def __init__(self):
        self.root = TNode()


def tree(data, label):
    tree2 = Tree()
    structure_tree(data, tree2.root, label)
    return tree2


def structure_tree(data, parent, label):
    best_gini, list = get_best_gini(data, label)

    left_list = list[0].drop(columns=best_gini.type, axis=1)
    right_list = list[1].drop(columns=best_gini.type, axis=1)

    if parent is None:
        parent = TNode()

    parent.left_list = left_list
    parent.right_list = right_list

    parent.type = best_gini.type
    parent.type_value = best_gini.type_value

    parent.left = TNode()
    parent.left.type = "结果"
    parent.left.type_value = "是"
    parent.right = TNode()
    parent.right.type = "结果"
    parent.right.type_value = "否"

    if len(left_list) is not 0 and len(left_list.columns) > 1:
        structure_tree(data=left_list, parent=parent.left, label=len(left_list.columns) - 1)
    if len(right_list) is not 0 and len(right_list.columns) > 1:
        structure_tree(data=right_list, parent=parent.right, label=len(right_list.columns) - 1)


def get_best_gini(data, label):
    row, col = data.shape
    gini_value_list = []
    for col_index in range(col):
        if col_index is label:
            continue
        else:
            # print("structure_tree : ", len(data), col_index)
            type_value_list = get_typeValueList_by_type(data, col_index)
            for type_value in type_value_list:
                ginidate = GiniData()
                ginidate.col_index = col_index
                ginidate.type = data.columns[col_index]
                ginidate.type_value = type_value
                ginidate.gini_rate = gini_by_type(data, col_index, type_value, label)
                gini_value_list.append(ginidate)

    gini_value_list.sort(key=operator.attrgetter('gini_rate'))

    # for gini in gini_value_list:
    #     print(gini.type, gini.type_value, gini.gini_rate)

    gini_best = gini_value_list[len(gini_value_list) - 1]
    return gini_best, get_typeList_by_type_and_typeValue(data, gini_best.col_index, gini_best.type_value)


def gini(data, label):
    """
    获取label类别下的gini指数
    :param data:
    :param label:
    :return:
    """
    row, col = data.shape
    label_type_list = get_typeValueList_by_type(data, label)

    label_type_rate = 0.0
    for type_value in label_type_list:
        label_type_list, demo = get_typeList_by_type_and_typeValue(data, label, type_value)

        label_type_rate += ((len(label_type_list) / row) ** 2)
        # print(len(label_type_list), label_type_rate)

    # print("gini return", 1 - label_type_rate)
    return 1 - label_type_rate


def gini_by_type(data, type1, type1_value, label):
    """
    获取在label类别下 特征为type1 特征值为type1_value 的gini指数
    :param data:
    :param type1:
    :param type1_value:
    :param label:
    :return:
    """
    row, col = data.shape  # D
    type_list, type_list_comple = get_typeList_by_type_and_typeValue(data, type1, type1_value)  # D1,D1的互补

    value = len(type_list) / row * gini(type_list, label) + len(type_list_comple) / row * gini(
        type_list_comple, label)
    # print("gini_by_type: type1 :", type1, ",type1_value :", type1_value, ",gini:", value)
    return value


def get_typeValueList_by_type(data, type1):
    """
    获取某个 type 下 typeValueList

    :param data:
    :param type1:
    :return:
    """
    type_list = []
    m, n = data.shape
    for row in range(m):
        if len(type_list) == 0:
            # print("get_typeValueList_by_type", len(data), type(data), type1, data.iloc[row, type1])
            type_list.append(data.iloc[row, type1])
        else:
            tag = True
            for temp in type_list:
                if data.iloc[row, type1] == temp:
                    tag = False
            if tag:
                type_list.append(data.iloc[row, type1])
    # print("get_typeValueList_by_type : type_list ", type_list)
    return type_list


def get_typeList_by_type_and_typeValue(data, type1, type_value, type2=None, type2_value=None):
    """
    获取 在type2特征 type2——value下的 特征为type1 特征值为type——value的 一个dataFrame 和他的互补dataFrame
    :param data:
    :param type1:
    :param type_value:
    :param type2:
    :param type2_value:
    :return:
    """
    # print(data.columns[type1])
    if type2:
        type_matriloc = data[(data[data.columns[type1]] == type_value) & (data[data.columns[type2]] == type2_value)]
        type_matriloc_comple = data[
            (data[data.columns[type1]] != type_value) & (data[data.columns[type2]] == type2_value)]
    else:
        type_matriloc = data[data[data.columns[type1]] == type_value]
        type_matriloc_comple = data[data[data.columns[type1]] != type_value]

    # print("get_typeList_by_type_and_typeValue: : ", type1, type2, len(type_matriloc), len(type_matriloc_comple))

    return type_matriloc, type_matriloc_comple
