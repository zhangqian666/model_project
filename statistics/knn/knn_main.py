# encoding = utf-8

from statistics.knn.kd_tree import KDTree, Node
import pandas as pd

# df = pd.DataFrame([
#     {"外貌分数": 1, "能力指数": 2, "身份": "英雄"},
#     {"外貌分数": 3, "能力指数": 4, "身份": "狗熊"},
#     {"外貌分数": 5, "能力指数": 1, "身份": "狗熊"},
#     {"外貌分数": 7, "能力指数": 8, "身份": "狗熊"},
#     {"外貌分数": 0, "能力指数": 1, "身份": "英雄"},
#     {"外貌分数": 2, "能力指数": 4, "身份": "英雄"},
#     {"外貌分数": 2, "能力指数": 2, "身份": "狗熊"},
#     {"外貌分数": 4, "能力指数": 2, "身份": "狗熊"},
#     {"外貌分数": 1, "能力指数": 5, "身份": "狗熊"},
#     {"外貌分数": 13, "能力指数": 42, "身份": "英雄"},
#     {"外貌分数": 32, "能力指数": 12, "身份": "英雄"},
# ])
# df.to_csv("./train_data.csv", index=True)

read_data = pd.read_csv("./train_data.csv", usecols=["外貌分数", "能力指数", "身份"])

kd = KDTree(read_data[["外貌分数", "能力指数"]], read_data["身份"])

print("打印 新建的kd 树列表")
kd.pre_order(kd.root)

search_node = Node(data=[5, 2])
list_result = kd.search_data_by_k(search_node, 7)

print("\nk list : \n")
if len(list_result) > 0:
    for item in list_result:
        print("node : {0}   ,between node len : {1}".format(item, kd.len_between_node(item, search_node)))
