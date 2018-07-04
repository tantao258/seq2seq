"""
本篇代码将实现一个基础版的Seq2Seq，输入一个单词（字母序列），模型将返回一个对字母排序后的“单词”。
"""

from model import *
from utils import *

#超参数
epoch = 30
batch_size = 128
lstm_size = 50
n_layers = 2
encoder_embedding_size = 15
decoder_embedding_size = 15
learning_rate = 0.001

#创建data对象
data = Data()

#创建对象
seq2seq = basic_seq2seq(batch_size=batch_size,
                        learning_rate=learning_rate,
                        data=data,
                        n_layers=n_layers,
                        lstm_size=lstm_size,
                        encoder_embedding_size=encoder_embedding_size,
                        decoder_embedding_size=decoder_embedding_size)
seq2seq.train()
seq2seq.prediction()

