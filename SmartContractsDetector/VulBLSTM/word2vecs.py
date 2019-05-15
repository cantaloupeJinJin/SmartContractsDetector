# encoding: utf-8
'''
@author: jinjin
@contact: cantaloupejinjin@gmail.com
@file: word2vecs.py
@time: 2019/4/14 20:24
'''
import logging
import gensim
from gensim.models import word2vec
import multiprocessing
# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 直接用gemsim提供的API去读取txt文件，读取文件的API有LineSentence 和 Text8Corpus, PathLineSentences等。
sentences = word2vec.LineSentence("rawData/wordEmbdiing.txt")

# 训练模型，词向量的长度设置为200， 迭代次数为8，采用skip-gram模型，模型保存为bin格式，sg=0表示使用的是CBOW
#model = gensim.models.Word2Vec(sentences, size=200, sg=1, iter=8)
model = gensim.models.Word2Vec(sentences,size = 200,sg=0, iter=8)
model.wv.save_word2vec_format("rawData/word2Vec" + ".bin", binary=True)
#model.wv.save_word2vec_format("rawData/word2Vec" + ".vector", binary=False)

# 加载bin格式的模型
#wordVec = gensim.models.KeyedVectors.load_word2vec_format("word2Vec.bin", binary=True)