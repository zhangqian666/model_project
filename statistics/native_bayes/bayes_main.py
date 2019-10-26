# encoding =utf-8

import pandas as pd
import numpy as np
import statistics.native_bayes.bayes as bayes

# bayes.train()

read_data = pd.read_csv("./train_data.csv", usecols=["颜色", "容量", "品牌", "价格", "能不能买"])

data_list = read_data[["颜色", "容量", "品牌", "价格", "能不能买"]]

# search = {"颜色": "蓝色", "容量": "1",
#           "品牌": "RoyalCopenhagen", "价格": 50}

search = ["蓝色", "2", "Starbucks", 420]

bayes.bayes(np.asarray(data_list), -1, search, 1)
