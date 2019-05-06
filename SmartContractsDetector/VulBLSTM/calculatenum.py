# encoding: utf-8
'''
@author: jinjin
@contact: cantaloupejinjin@gmail.com
@file: calculatenum.py
@time: 2019/5/5 21:26
'''
import csv


with open('preprocess/labeledTrain.csv', 'r',encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        num = str(row[1]).split(" ")
        print(len(num))
