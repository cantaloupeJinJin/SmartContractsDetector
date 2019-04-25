# encoding: utf-8
'''
@author: jinjin
@contact: cantaloupejinjin@gmail.com
@file: modelPredict.py
@time: 2019/4/25 15:21
'''
import os
import csv
import time
import datetime
import random
import json

import warnings
from collections import Counter
from math import sqrt

import gensim
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
warnings.filterwarnings("ignore")
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
#from BiLSTM import Dataset,TrainingConfig,ModelConfig,Config,BiLSTM
# sequenceLength = 600
# wordToIndex = {}
# '''
# 获取评论
# '''
# filePath = "preprocess/test1.csv"
# df = pd.read_csv(filePath)
# review = df["review"].to_list()

#
# reviewVec = Dataset._reviewProcess(review, sequenceLength, wordToIndex)
#model/Bi-LSTM/modelpb/saved_model.pb
from tensorflow.python.platform import gfile
pb_file_path = "model/Bi-LSTM/modelpb/saved_model.pb"
def restore_mode_pb(pb_file_path):
    sess = tf.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    # print(sess.run('inputX:0'))

if __name__ == '__main__':
    restore_mode_pb(pb_file_path)
