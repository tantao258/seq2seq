import os
import json
import pickle
import nltk
import numpy as np
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict

class DataProcess(object):
    def __init__(self, num_classes=5, filter_n=5):
        self.yelp_json_path = "./data/yelp_academic_dataset_review.json"
        self.yelp_data_path = self.yelp_json_path[0:-5] + "_data.pickle"
        self.vocab_path = self.yelp_json_path[0:-5] + "_vocab.pickle"
        self.max_sentence_in_doc = 30
        self.max_word_in_sentence = 30
        self.filter_n = filter_n            # 词频低于filter_n的词被去掉
        self.num_classes = num_classes

    def build_vocab(self):
        if os.path.exists(self.vocab_path):
            vocab_file = open(self.vocab_path, 'rb')
            vocab = pickle.load(vocab_file)
            print("load vocab finish!")

        else:
            # 记录每个单词及其出现的频率,构造词表
            word_freq = {}
            # 读取数据集，并进行分词，统计每个单词出现次数，保存在word freq中
            with open(self.yelp_json_path, "rb") as f:
                for line in f:
                    review = json.loads(line.decode('utf-8'))
                    words = WordPunctTokenizer().tokenize(review['text'])
                    for word in words:
                        if word in word_freq:
                            word_freq[word] += 1
                        else:
                            word_freq[word] =1

            # print(len(word_freq))

            # 构建vocabulary，并将出现次数小于n的单词全部去除，视为UNKNOW
            vocab = {}
            vocab['UNKNOW_TOKEN'] = 0
            i = 1
            for word, freq in (word_freq.items()):
                if freq > self.filter_n:
                    vocab[word] = i
                    i += 1

            # 保存词典
            with open(self.vocab_path, 'wb') as g:
                pickle.dump(vocab, g)

        return vocab, len(vocab)

    def data_load(self):
        doc_num = 229907  # documents数量

        if not os.path.exists(self.yelp_data_path):
            self.vocab, self.vocab_size = self.build_vocab()
            UNKNOWN = 0

            data_x = np.zeros([doc_num, self.max_sentence_in_doc, self.max_word_in_sentence])
            data_y = []

            # 将所有的评论文件都转化为30*30的索引矩阵，也就是每篇都有30个句子，每个句子有30个单词
            # 不够的补零，多余的删除，并保存到最终的数据集文件之中
            sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
            with open(self.yelp_json_path, "rb") as f:
                for line_index, line in enumerate(f):
                    review = json.loads(line.decode('utf-8'))
                    sentences = sent_tokenizer.tokenize(review['text'])
                    doc = np.zeros([self.max_sentence_in_doc, self.max_word_in_sentence])

                    for i, sentence in enumerate(sentences):
                        if i < self.max_sentence_in_doc:
                            word_to_index = np.zeros([self.max_word_in_sentence], dtype=int)
                            for j, word in enumerate(WordPunctTokenizer().tokenize(sentence)):
                                if j < self.max_word_in_sentence:
                                    word_to_index[j] = self.vocab.get(word, UNKNOWN)
                            doc[i] = word_to_index


                    data_x[line_index] = doc
                    label = int(review['stars'])
                    labels = [0] * self.num_classes
                    labels[label - 1] = 1
                    data_y.append(labels)

                pickle.dump((data_x, data_y), open(self.yelp_data_path, 'wb'))

        else:
            data_file = open(self.yelp_data_path, 'rb')
            data_x, data_y = pickle.load(data_file)


        length = len(data_x)
        train_x, dev_x = data_x[:int(length * 0.9)], data_x[int(length * 0.9) + 1:]
        train_y, dev_y = data_y[:int(length * 0.9)], data_y[int(length * 0.9) + 1:]

        return train_x, train_y, dev_x, dev_y


if __name__ == '__main__':
    data = DataProcess()
    train_x, train_y, dev_x, dev_y = data.data_load()
    print(train_x.shape)
    print(train_x[51, 0, 0])
