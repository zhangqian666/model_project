# encoding = utf-8

import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from statsmodels.tsa.api import VECM, VAR
import statsmodels.tsa.stattools as ts
from statsmodels.tsa.base.datetools import dates_from_str


def read_futures(file):
    futures_data = pd.read_csv(file,
                               usecols=["收盘价:螺纹指数", "收盘价:热卷指数", "指标名称"])
    quarterly = futures_data["指标名称"].astype(str)

    quarterly = dates_from_str(quarterly[2001:2373])

    futures_data = futures_data[2000:2373]

    futures_data = futures_data[["收盘价:螺纹指数", "收盘价:热卷指数"]]

    futures_data_luowen = futures_data["收盘价:螺纹指数"]

    futures_data_rejuan = futures_data["收盘价:热卷指数"]
    print("############### - 输出 luowen 的 数据 - #############")
    print(futures_data_luowen)

    print("############### - 输出 luowen 的 ADF - #############")
    adf(futures_data_luowen)
    print("############### - 输出 rejuan 的 数据 - #############")
    print(futures_data_luowen)
    print("############### - 输出 rejuan 的 ADF - #############")
    adf(futures_data_rejuan)

    futures_data_luowen_new = []
    futures_data_rejuan_new = []
    for i in range(2001, 2373, 1):
        futures_data_luowen_new.append(futures_data_luowen[i] / futures_data_luowen[i - 1])

    for i in range(2001, 2373, 1):
        futures_data_rejuan_new.append(futures_data_rejuan[i] / futures_data_rejuan[i - 1])

    return DataFrame({"luowen": futures_data_luowen_new, "rejuan": futures_data_rejuan_new},
                     columns=["luowen", "rejuan"]), quarterly


def var(data):
    var_model = VAR(data)
    var_results = var_model.fit(2)

    print(var_results.summary())



def adf(data):
    adfResult = ts.adfuller(data)
    output = pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used",
                                 "Critical Value(1%)", "Critical Value(5%)", "Critical Value(10%)"],
                          columns=['value'])
    output['value']['Test Statistic Value'] = adfResult[0]
    output['value']['p-value'] = adfResult[1]
    output['value']['Lags Used'] = adfResult[2]
    output['value']['Number of Observations Used'] = adfResult[3]
    output['value']['Critical Value(1%)'] = adfResult[4]['1%']
    output['value']['Critical Value(5%)'] = adfResult[4]['5%']
    output['value']['Critical Value(10%)'] = adfResult[4]['10%']
    print(output)


def vecm(data):
    vecm_model = VECM(data)
    vecm_results = vecm_model.fit()
    print(vecm_results.summary())


if __name__ == "__main__":
    read_data, quarterly = read_futures("futures_data.csv")
    read_data.index = pd.DatetimeIndex(quarterly)
    print("############### - 输出 vsc 文件中的 今天的数据除以昨天的数据 - #############")
    print(read_data)

    print("############ - 输出 index - ################")
    data = np.log(read_data).diff().dropna()
    print(quarterly)
    print("########### - 输出log后的数据 - #################")
    print(data)
    print("########## - 执行VECM- ##################")
    vecm(data)

    var(data)

