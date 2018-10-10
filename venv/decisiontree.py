# coding=utf-8
from numpy import *
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from math import log
import operator
import json
import os
import sys


# 按照某一列的值将文件分组
def splitgroupby():
    # 读取samrt属性文件 在文件中手动加入了列名
    # df=pd.read_csv('D:\\sublimedemo\\dataSet\\Disk_SMART_dataset.csv')
    # 读取西瓜文件
    df = pd.read_csv('D:\\sublimedemo\\dataSet\\xigua.csv', encoding='utf-8')
    # 读入文件的行和列 限制 在[]中[start：stop]不包含stop位置
    data = df.values[:2, :3].tolist()
    # 读取列名
    labels = df.columns.values[0:14].tolist()
    # 采用json的形式输出可以防止中文不输出
    print(json.dumps(labels, ensure_ascii=False, indent=4))
    # print labels[2]
    # 按照某一属性将数据集分组
    groups = df.groupby([df[u'密度'] > 0.5, df[u'好瓜'], df[u'纹理']])
    # 夺取当前工作路径
    print(os.getcwd())
    # print groups
    i = 0
    # head() tail()默认都是查看五行数据 descibe可以描述df 平均值 等特性
    # print df.describe()
    # 将分组的数据集写入文件
    for group in groups:
        # group.to_csv('D:\\sublimedemo\\dataSet\\' + i + '.csv', index=False, encoding='utf-8')
        # group[1].to_csv('D:\\sublimedemo\\dataSet\\xiguawenli' + str(i) + \
        # 	'.csv', index = False, float_format = '%.3f', encoding = 'utf-8')
        # i+=1
        print(group[1])


# 在每一个分组的文件中选择分组的第一行数据
def choose_firstraw():
    files = []
    datas = []
    for i in range(0, 23394):
        filenames = 'D:\\sublimedemo\\dataSet\\split-Disk_SMART_dataset-bycolumn1\\Disk_SMART_dataset' \
                    + str(i) + '.csv'
        files.append(filenames)
    for filename in files:
        df = pd.read_csv(filename, encoding='utf-8')
        data = df.values[0:1].tolist()
        datas.extend(data)
    # 将List数据类型转换为DataFrame
    df1 = DataFrame(datas)
    df1.to_csv('D:\\sublimedemo\\dataSet\\split-Disk_SMART_dataset-firstraw' \
               + '.csv', index=False, float_format='%.6f', encoding='utf-8')


# 将dataset数据类型进行变换
def change_datatype():
    filename = 'D:\\sublimedemo\\dataSet\\split-Disk_SMART_dataset-firstraw.csv'
    df = pd.read_csv(filename, encoding='utf-8')
    print (df.dtypes)
    labels = df.columns.values[:].tolist()
    df['0'] = df['0'].astype('int')
    df.to_csv('D:\\sublimedemo\\dataSet\\split-Disk_SMART_dataset-firstraw1' \
              + '.csv', index=False, float_format='%.6f', encoding='utf-8')
    return df, labels


# 计算数据集的熵值
def jisuanEnt(dataSet):
    labelcounts = {}
    num = len(dataSet)
    for featVal in dataSet:
        currentLabel = featVal[0]
        if currentLabel not in labelcounts.keys():
            labelcounts[currentLabel] = 0
        labelcounts[currentLabel] += 1
    ent = 0.0
    for key in labelcounts:
        prob = float(labelcounts[key]) / num
        ent -= prob * log(prob, 2)
    return ent


# 对决策树的离散的属性集进行划分 对于Smart属性集划分 应用不到
def splitDataSet(dataSet, axis, value):
    resultSet = []
    for record in dataSet:
        if record[axis] == value:
            templist = record[:axis]
            templist.extend(record[axis + 1:])
            resultSet.append(templist)
    return resultSet


# 设定对连续的数据集可以进行划分的方法
def splitcontinuoousDataSet(dataSet, axis, value, dirction):
    resultSet = []
    for record in dataSet:
        if dirction == 0:
            if record[axis] >= value:
                templist = record[:axis]
                templist.extend(record[axis + 1:])
                resultSet.append(templist)
        else:
            if record[axis] < value:
                templist = record[:axis]
                templist.extend(record[axis + 1:])
                resultSet.append(templist)
    return resultSet


# 寻找进行最优划分的属性
def chooseBestFeatureToSplit(dataSet, labels):
    numFeature = len(dataSet[0])
    baseEntroy = jisuanEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(1, numFeature):
        # print '循环是' + str(i)
        featureList = [exmple[i] for exmple in dataSet]
        splitList = [float(x) / 10 for x in range(-9, 10)]
        bestSplitEntropy = 100000
        slen = len(splitList)
        for j in range(slen):
            # print '划分次数' + str(j)
            newEny = 0.0
            value = splitList[j]
            subDataSet0 = splitcontinuoousDataSet(dataSet, i, value, 0)
            subDataSet1 = splitcontinuoousDataSet(dataSet, i, value, 1)
            prob0 = float(len(subDataSet0)) / len(data)
            newEny += prob0 * jisuanEnt(subDataSet0)
            prob1 = float(len(subDataSet1)) / len(data)
            newEny += prob1 * jisuanEnt(subDataSet1)
            if newEny < bestSplitEntropy:
                bestSplitEntropy = newEny
                bestSplit = j
        bestSplitDict[labels[i - 1]] = splitList[bestSplit]
        infoGain = baseEntroy - bestSplitEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i - 1
    bestSplitValue = bestSplitDict[labels[bestFeature]]
    labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
    for i in range(shape(dataSet)[0]):
        if dataSet[i][bestFeature] <= bestSplitValue:
            dataSet[i][bestFeature] = 1
        else:
            dataSet[i][bestFeature] = 0
    return bestFeature


def createTree(dataSet, labels, data_full, labels_full):
    # 将dataSet的第一列取出
    classList = [example[0] for example in dataSet]
    # 如果行数过少不再划分
    if len(classList) <= 6:
        return majorityCnt(classList)
    # 如果类别相同停止划分
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 遍历完所有特征时返回出现次数最多的
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    # 选择信息增益最大的属性
    bestFeat = chooseBestFeatureToSplit(dataSet, labels)
    bestFeatLabel = labels[bestFeat]
    # 采用递归的方法创建决策树
    myTree = {bestFeatLabel: {}}
    # 将bestFeat的所有取值找出
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    # if type(dataSet[0][bestFeat + 1]).__name__ == 'str':
    # currentlabel = labels_full.index(labels[bestFeat])
    # featValuesFull = [example[currentlabel] for example in data_full]
    # uniqueValsFull = set(featValuesFull)
    del (labels[bestFeat])
    # 针对bestFeat的每个取值，划分出一个子树。
    for value in uniqueVals:
        subLabels = labels[:]
        # if type(dataSet[0][bestFeat + 1]).__name__ == 'str':
        # uniqueValsFull.remove(value)
        myTree[bestFeatLabel][value] = createTree(splitDataSet \
                                                      (dataSet, bestFeat, value), subLabels, data_full, labels_full)

    # if type(dataSet[0][bestFeat]).__name__=='str':
    # 	for value in uniqueValsFull:
    # 		myTree[bestFeatLabel][value]=majorityCnt(classList)
    return myTree


# 特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCnt(classList):
    classcount = {}
    for record in classList:
        if record not in classcount.keys():
            classcount[record] = 0
        classcount[record] += 1
    return max(classcount)


def shuzhiqujian(data):
    t = []
    num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for j in range(0, 13):
        num = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(0, len(data)):
            if data[i][j] >= -1 and data[i][j] <= -0.9:
                num[0] += 1
            elif data[i][j] > -0.9 and data[i][j] <= -0.8:
                num[1] += 1
            elif data[i][j] > -0.8 and data[i][j] <= -0.7:
                num[2] += 1
            elif data[i][j] > -0.7 and data[i][j] <= -0.6:
                num[3] += 1
            elif data[i][j] > -0.6 and data[i][j] <= -0.5:
                num[4] += 1
            elif data[i][j] > -0.5 and data[i][j] <= -0.4:
                num[5] += 1
            elif data[i][j] > -0.4 and data[i][j] <= -0.3:
                num[6] += 1
            elif data[i][j] > -0.2 and data[i][j] <= -0.1:
                num[7] += 1
            elif data[i][j] > -0.1 and data[i][j] <= 0:
                num[8] += 1
            elif data[i][j] > 0 and data[i][j] <= 0.1:
                num[9] += 1
            elif data[i][j] > 0.1 and data[i][j] <= 0.2:
                num[10] += 1
            elif data[i][j] > 0.2 and data[i][j] <= 0.3:
                num[11] += 1
            elif data[i][j] > 0.3 and data[i][j] <= 0.4:
                num[12] += 1
            elif data[i][j] > 0.4 and data[i][j] <= 0.5:
                num[13] += 1
            elif data[i][j] > 0.5 and data[i][j] <= 0.6:
                num[14] += 1
            elif data[i][j] > 0.6 and data[i][j] <= 0.7:
                num[15] += 1
            elif data[i][j] > 0.7 and data[i][j] <= 0.8:
                num[16] += 1
            elif data[i][j] > 0.8 and data[i][j] <= 0.9:
                num[17] += 1
            elif data[i][j] > 0.9 and data[i][j] <= 1:
                num[18] += 1
        t.append(num)
    print(len(t))

    for i in range(len(t)):
        print(t[i], i)
if __name__ == '__main__':
    df = pd.read_csv('/home/zzk/test/dataSet/split-Disk_SMART_dataset-firstraw.csv')
    data = df.values[0:10, 1:].tolist()
    data2 = df.values[500:1000, 1:].tolist()
    data.extend(data2)
    data_full = data[:]
    # labels = df.columns[2:].tolist()
    columnsName = ['NO', 'Type', 'RRR', 'SUP', 'RSC', 'SER', 'POH', 'RUE', 'HFW', 'TC', 'HR', 'CPSC1', 'RSC', 'CPSC2']
    df.columns = columnsName
    labels = df.columns[2:].tolist()
    labels_full = columnsName
    # 创建决策树的语句
    # createTree(data,labels,data_full,labels_full)
    myTree = createTree(data, labels, data_full, labels_full)
    print(json.dumps(myTree, ensure_ascii=False, indent=4))

    print('ok')
