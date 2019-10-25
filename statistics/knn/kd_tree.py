# encoding = utf-8

import numpy as np
from numpy.linalg import norm


class Node:

    def __init__(self, left=None, right=None, parent=None, data=None, axis=None, label=None):
        """
        构造函数
        :param left 左孩子节点
        :param right 右孩子节点
        :param parent 父节点
        :param data 数据
        :param axis 分割点
        :param label 标记

        """
        self.left = left
        self.right = right
        self.parent = parent

        self.data = data
        self.axis = axis
        self.label = label

    def __str__(self):
        return "{0},{1},{2}".format(self.data, self.axis, self.label)


class KDTree:

    def __init__(self, x, y=Node):
        """
        构造函数

        :param x: X的特征集 n_simples *n_features
        :param y: y的label值列表
        """
        self.root = None
        self.y_valid = False if y is None else True
        self.__create_kd_tree(x, y)

    def __create_kd_tree(self, x, y):
        """
        创建kd树

        :param x: X的特征集 n_simples *n_features
        """

        if y is not None:
            xy = np.hstack((np.array(x), np.array([y]).T))
        print(xy)
        self.root = self.__create_node(xy, 0)

    def __create_node(self, xy, axis, parent=None):
        """
        创建kd树根结点

        :param xy:  X 和 y的数据集合
        :param axis: 切分轴
        :return: Node 根结点
        """
        n_samples = np.shape(xy)[0]  # 例子的数量
        k_dimensions = np.shape(xy[:, :-1])[-1]  # X特征集 的维度

        if n_samples == 0:
            return None
        mid = n_samples >> 1

        next_axis = (axis + 1) % k_dimensions

        print(xy, axis)
        sorted_xy = self.__sort_xy_data(xy, axis)  # 进行排序（按照axis排序）

        left_data = sorted_xy[:mid]
        right_data = sorted_xy[mid + 1:]

        root = Node(data=sorted_xy[mid, :-1], label=sorted_xy[mid, -1], axis=axis, parent=parent)
        root.left = self.__create_node(left_data, next_axis, root)
        root.right = self.__create_node(right_data, next_axis, root)
        return root

    def __sort_xy_data(self, xy, axis):
        """
        冒泡排序

        :param xy:
        :param axis:
        :return:
        """
        sort_data = xy
        m, n = np.shape(sort_data)

        for i in range(m - 1):
            for j in range(m - i - 1):
                if sort_data[j, axis] > sort_data[j + 1, axis]:
                    sort_data[[j, j + 1], :] = sort_data[[j + 1, j], :]
        return sort_data

    def pre_order(self, node):
        """
        前序查询

        :param node:
        :return:
        """
        if node is not None:
            print(node.data, node.label)
            self.pre_order(node.left)
            self.pre_order(node.right)

    def search_data_by_k(self, search_node, k):
        """
        查询k近邻node

        :param search_node: 查找目标
        :param k: k近邻 的k个数据
        :return: nearest_list k个最近的数据
        """
        nearest_list = []
        self.__search_nearest_node_list(nearest_list, self.root, search_node, k)
        return nearest_list

    def __search_nearest_node_list(self, nearest_list, current_node, search_node, k):
        """
        递归搜索方法

        :param nearest_list:
        :param current_node:
        :param search_node:
        :param k:
        :return:
        """
        if current_node is None:
            return

        current_node_len = self.len_between_node(current_node, search_node)
        nearest_list_longest_len = None

        if len(nearest_list) > 0:
            nearest_list_longest_len = self.len_between_node(nearest_list[0], search_node)

        if len(nearest_list) < k:
            print(current_node, "-------------------------------append to list")
            nearest_list.append(current_node)
        elif current_node_len < nearest_list_longest_len:
            print(current_node, "-------------------------------change from list 第0个")
            nearest_list[0] = current_node

        self.__sort_list_by_len(nearest_list, search_node)

        if int(current_node.data[current_node.axis]) > search_node.data[current_node.axis]:
            if current_node.left is not None:
                self.__search_nearest_node_list(nearest_list, current_node.left, search_node, k)
            else:
                return
        else:
            if current_node.right is not None:
                self.__search_nearest_node_list(nearest_list, current_node.right, search_node, k)
            else:
                return

        if (current_node.left is not None) or (current_node.rigth is not None):

            if int(current_node.data[current_node.axis]) < search_node.data[current_node.axis]:
                self.__search_nearest_node_list(nearest_list, current_node.left, search_node, k)
            else:
                self.__search_nearest_node_list(nearest_list, current_node.right, search_node, k)

    def __sort_list_by_len(self, nearest_list, search_node):
        """
        存储k个最近的数据

        :param nearest_list:
        :param search_node:
        :return:
        """
        for i in range(len(nearest_list)):
            for j in range(len(nearest_list) - i - 1):
                if self.len_between_node(nearest_list[j], search_node) < self.len_between_node(nearest_list[j + 1],
                                                                                               search_node):
                    nearest_list[j], nearest_list[j + 1] = nearest_list[j + 1], nearest_list[j]

        print("==========sort start ===========")
        for i in nearest_list:
            print(self.len_between_node(i, search_node), "--")
        print("==========sort end ===========")

    def len_between_node(self, node1, node2):
        """
        获取两个node之间的距离

        :param node1:
        :param node2:
        :return:
        """

        # print("---", node1, node2, "---")
        data = node1.data
        new_data = [int(data[0]), int(data[1])]
        # print(new_data, node2.data)
        return norm(np.asarray(new_data) - np.asarray(node2.data), ord=2)
