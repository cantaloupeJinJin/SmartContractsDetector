# encoding: utf-8
'''
@author: jinjin
@contact: cantaloupejinjin@gmail.com
@file: expanddata.py
@time: 2019/5/10 11:13
'''
import pandas as pd
from  imblearn.over_sampling import SMOTE
filePath = "preprocess/test1.csv"
df = pd.read_csv(filePath,names=['sentiment','review'])
x = df["review"].tolist()  #x
y = df["sentiment"].tolist()#y
groupby_data_orginal = df.groupby('sentiment').count()
print("print x", x)
print("print y", y)
print(groupby_data_orginal)#打印输出原始数据集样本分类分布

# 使用SMOTE方法进行过抽样处理
model_smote = SMOTE() # 建立SMOTE模型对象
x_smote_resampled, y_smote_resampled = model_smote.fit_sample(x,y) # 输入数据并作过抽样处理
print("上一步没问题")
x_smote_resampled = pd.DataFrame(x_smote_resampled, columns=['review']) # 将数据转换为数据框并命名列名
y_smote_resampled = pd.DataFrame(y_smote_resampled,columns=['sentiment']) # 将数据转换为数据框并命名列名
smote_resampled = pd.concat([x_smote_resampled, y_smote_resampled],axis=1) # 按列合并数据框
groupby_data_smote = smote_resampled.groupby('sentiment').count() # 对label做分类汇总
print (groupby_data_smote) # 打印输出经过SMOTE处理后的数据集样本分类分布
