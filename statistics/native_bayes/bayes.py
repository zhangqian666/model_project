# encoding =utf-8
import random


def bayes(data_list, label_name, search, lamda):
    label_value_list = search_type(data_list, label_name)
    p = {}

    for label_value in label_value_list:
        pxy = 1
        for key in range(len(search)):
            label_num = search_number(data_list, label_name, label_value) + lamda * len(search_type(data_list, key))

            pxy *= (search_number(data_list, key, search[key], label_value=label_value) + lamda) / label_num
            print(pxy, label_num)

        pk = (search_number(data_list, label_name, label_value) + lamda) / (
                len(data_list) + len(search_type(data_list, label_name)) * lamda)

        p[label_value] = pxy * pk

    print(p)


def search_type(data_list, type):
    columns = ["颜色", "容量", "品牌", "价格", "能不能买"]
    type_list = []
    for item in data_list:
        if len(type_list) == 0:
            type_list.append(item[type])
        else:
            flag = False
            for i in type_list:
                if item[type] == i:
                    flag = True

            if not flag:
                type_list.append(item[type])

    # print(columns[type], len(type_list))
    return type_list


def search_number(data_list, type, value, label_value=None):
    columns = ["颜色", "容量", "品牌", "价格", "能不能买"]
    count = 0
    for item in data_list:
        if label_value is not None:
            print(item[-1], label_value, item[type], value)
            if (item[-1] == label_value) and (item[type] == value):
                # print(item[-1], label_value, item[type], value)
                count += 1
        else:
            if item[type] == value:
                count += 1
    if type != -1:
        print(len(data_list), columns[type], value, label_value, count)
    return count


def train():
    yanse = ["红色", "蓝色", "白色", "紫色"]
    rongliang = ["1", "2", "3", "4"]
    pinpai = ["Starbucks", "RoyalCopenhagen", "UCC", "Maxwell", "Nescafe"]
    jiage = [100, 50, 230, 21, 420]

    list = []

    for i in range(1000):
        data = {"颜色": yanse[random.randint(0, len(yanse) - 1)], "容量": rongliang[random.randint(0, len(rongliang) - 1)],
                "品牌": pinpai[random.randint(0, len(pinpai) - 1)], "价格": jiage[random.randint(0, len(jiage) - 1)],
                "能不能买": random.randint(0, 1)}
        list.append(data)

    for item in list:
        print(item)

    df = pd.DataFrame(list, columns=["颜色", "容量", "品牌", "价格", "能不能买"])

    df.to_csv("./train_data.csv", columns=["颜色", "容量", "品牌", "价格", "能不能买"], index=True)
