# encoding = utf-8

from statistics.knn.kd_tree import KDTree, Node

kd = KDTree([[1, 2], [3, 4], [5, 1], [7, 8], [0, 1], [2, 4], [2, 2], [4, 2], [1, 5], [13, 42], [32, 12]],
            ["英雄", "狗熊", "狗熊", "狗熊", "英雄", "英雄", "狗熊", "狗熊", "狗熊", "英雄", "英雄"])

print("打印 新建的kd 树列表")
kd.pre_order(kd.root)

search_node = Node(data=[5, 2])
list_result = kd.search_data_by_k(search_node, 7)

print("\nk list : \n")
if len(list_result) > 0:
    for item in list_result:
        print("node : {0}   ,between node len : {1}".format(item, kd.len_between_node(item, search_node)))
