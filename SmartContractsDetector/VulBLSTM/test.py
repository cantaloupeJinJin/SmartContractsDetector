import os
import csv
import time
import datetime
import random
import json
import keras
import warnings
from collections import Counter
from math import sqrt

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score

dataSource = "preprocess/labeledTrain2.csv"
rate = 1  # 训练集的比例


def _readData(filePath):
    """
    从csv文件中读取数据集
    """

    df = pd.read_csv(filePath)
    labels = df["sentiment"].tolist()
    reviews = df["review"].tolist()

    return reviews, labels

def _genTrainEvalData(x, y, rate):
    """
    生成训练集和验证集
    """

    reviews = []
    labels = []

    trainIndex = int(len(x) * rate)

    trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
    trainLabels = np.array(labels[:trainIndex], dtype="float32")

    evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
    evalLabels = np.array(labels[trainIndex:], dtype="float32")

    return trainReviews, trainLabels, evalReviews, evalLabels

reviews, labels = _readData(dataSource)
print(reviews)
print(labels)
trainReviews, trainLabels, evalReviews, evalLabels = _genTrainEvalData(reviews, labels, rate)

def nextBatch(x, y):
    """
    生成batch数据集，用生成器的方式输出
    """

    perm = np.arange(len(x))
    np.random.shuffle(perm)
    x = x[perm]
    y = y[perm]
    print(x)
    return x, y

x, y = nextBatch(trainReviews, trainLabels)

def create_csv():
    path = "preprocess/write_test.csv"
    with open(path,'w') as f:
        csv_write = csv.writer(f)
        csv_head = ["sentiment","review"]
        csv_write.writerow(csv_head)
        csv_write.writerows([x ,y])

if __name__ == "__main__":
    create_csv()