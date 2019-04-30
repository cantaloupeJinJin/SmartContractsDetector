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
#import tensorflow as tf
from tensorflow.python.platform import gfile
pb_file_path = "model/Bi-LSTM/modelpb/saved_model.pb"
#pb_file_path = "model/Bi-LSTM/modelpb/frozen_inference_graph"
def restore_mode_pb(pb_file_path):
    sess = tf.Session()

    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    print(sess.run('b:0'))

    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')

    op = sess.graph.get_tensor_by_name('op_to_store:0')

    ret = sess.run(op, {input_x: 5, input_y: 5})
    print(ret)

if __name__ == "__main__":
    restore_mode_pb(pb_file_path)