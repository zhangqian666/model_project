# encoding = utf-8

import pandas as  pd
import statistics.decistion_tree.decision_tree as dt

read_data = pd.read_csv("./train_data2.csv", usecols=["颜色", "容量", "品牌", "价格", "能不能买"])

reTree = dt.tree(read_data, len(read_data.columns) - 1)


def preTree(node):
    if node:
        print("===", node, "===")
        preTree(node.left)
        preTree(node.right)


preTree(reTree.root)
