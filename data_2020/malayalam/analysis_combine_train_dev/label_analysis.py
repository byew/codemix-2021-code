import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# def hydata_train():
#     # 读取文件
#     pr = pd.read_csv("train.tsv", sep='\t')
#     # print(pr)
#     # a = []
#     man = 0
#     woman = 0
#     # 统计男女比例
#     a = pr['class']
#
#     for sex in a:  # 从XB列读取数据
#         if sex == 1:
#             man += 1
#         elif sex == 0:
#             woman += 1
#
#
#     print(man)
#
#     # 绘制饼状图
#     labels = ['label_1', 'label_0']
#     # 绘图显示的标签
#     values = [man, woman]
#     colors = ['y', 'r']
#     explode = [0, 0.1]
#     # 旋转角度
#     plt.title("Label distribution in train set", fontsize=25)
#     plt.pie(values, labels=labels, explode=explode, colors=colors,
#         startangle=180,
#         shadow=True, autopct='%1.1f%%')
#     plt.axis('equal')
#     plt.show()


def hydata_test():
    # 读取文件
    pr = pd.read_csv("dev.tsv", sep='\t')
    # print(pr)
    # a = []
    man = 0
    woman = 0
    # 统计男女比例
    a = pr['class']

    for sex in a:  # 从XB列读取数据
        if sex == 1:
            man += 1
        elif sex == 0:
            woman += 1


    print(man)

    # 绘制饼状图
    labels = ['label_1', 'label_0']
    # 绘图显示的标签
    values = [man, woman]
    colors = ['y', 'r']
    explode = [0, 0.1]
    # 旋转角度
    plt.title("Label distribution in train set", fontsize=25)
    plt.pie(values, labels=labels, explode=explode, colors=colors,
        startangle=180,
        shadow=True, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()

if __name__=="__main__":
    # hydata_train()
    hydata_test()