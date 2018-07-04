import tensorflow as tf
import time
import os
import numpy as np
from tensorflow.contrib import rnn
from tensorflow.contrib import layers

def length(sequences):
    used = tf.sign(tf.reduce_max(tf.abs(sequences), reduction_indices=2))
    seq_len = tf.reduce_sum(used, reduction_indices=1)
    return tf.cast(seq_len, tf.int32)

class HAN():

    def __init__(self, learning_rate,
                 batch_size,
                 grad_clip,
                 vocab_size,
                 num_classes,
                 embedding_size=200,
                 hidden_size=50
                 ):

        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.learning_rate = learning_rate
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.build_inputs()
        self.word2vec()
        self.sent2vec()
        self.doc2vec()
        self.build_classifier()
        self.build_loss()
        self.build_optimizer()
        self.build_accuracy()



    def build_inputs(self):
        with tf.name_scope('inputs'):
            self.max_sentence_num = tf.placeholder(tf.int32, name='max_sentence_num')
            self.max_sentence_length = tf.placeholder(tf.int32, name='max_sentence_length')
            #x的shape为[batch_size, 句子数， 句子长度(单词个数)]，但是每个样本的数据都不一样，，所以这里指定为空
            #y的shape为[batch_size, num_classes]
            self.input_x = tf.placeholder(tf.int32, [None, None, None], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')

    def word2vec(self):
        with tf.name_scope("embedding"):
            embedding_matrix = tf.Variable(initial_value=tf.truncated_normal((self.vocab_size, self.embedding_size)))
            # shape为[batch_size, sent_in_doc, word_in_sent, embedding_size]
            self.word_embedded = tf.nn.embedding_lookup(embedding_matrix, self.input_x)

    def sent2vec(self):
        with tf.name_scope("sent2vec"):
            # GRU的输入tensor是[batch_size, max_time, ...].在构造句子向量时max_time应该是每个句子的长度，所以这里将
            # batch_size * sent_in_doc当做是batch_size.这样一来，每个GRU的cell处理的都是一个单词的词向量
            # 并最终将一句话中的所有单词的词向量融合（Attention）在一起形成句子向量

            # word_embedded.shape为[batch_size*sent_in_doc, word_in_sent, embedding_size]
            word_embedded = tf.reshape(self.word_embedded, [-1, self.max_sentence_length, self.embedding_size])

            # word_encoder
            # word_encoded.shape为[batch_size*sent_in_doc, word_in_sent, hidden_size*2]
            word_encoded = self.BidirectionalGRUEncoder(word_embedded, name='word_encoder')

            # word_attention
            # sent_vec.shape为[batch_size*sent_in_doc, hidden_size*2]
            self.sent_vec = self.AttentionLayer(word_encoded, name='word_attention')

    def doc2vec(self):
        with tf.name_scope("doc2vec"):
            sent_vec = tf.reshape(self.sent_vec, [-1, self.max_sentence_num, self.hidden_size * 2])

            # sentence_encoder
            # shape为[batch_size, sent_in_doc, hidden_size*2]
            sent_encoded = self.BidirectionalGRUEncoder(sent_vec, name='sent_encoder')

            # shape为[batch_szie, hidden_szie*2]
            self.doc_vec = self.AttentionLayer(sent_encoded, name='sent_attention')

    def build_classifier(self):
        with tf.name_scope('doc_classification'):
            self.out = layers.fully_connected(inputs=self.doc_vec,
                                              num_outputs=self.num_classes,
                                              activation_fn=None)

    def build_loss(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,
                                                                           logits=self.out,
                                                                           name='loss'))
        self.loss_summary = tf.summary.scalar('loss', self.loss)

    def build_optimizer(self):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # RNN中常用的梯度截断，防止出现梯度过大难以求导的现象
            train_vars = tf.trainable_variables()
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, train_vars), self.grad_clip)
            grads_and_vars = tuple(zip(grads, train_vars))
            self.train_op = optimizer.apply_gradients(grads_and_vars)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    grad_summaries.append(grad_hist_summary)
            self.grad_summaries_merged = tf.summary.merge(grad_summaries)

    def build_accuracy(self):
        with tf.name_scope('accuracy'):
            predict = tf.argmax(self.out, axis=1, name='predict')
            label = tf.argmax(self.input_y, axis=1, name='label')
            self.acc = tf.reduce_mean(tf.cast(tf.equal(predict, label), tf.float32))

            # tensorboard
            self.acc_summary = tf.summary.scalar('accuracy', self.acc)

    def train(self, train_x, train_y, dev_x, dev_y, epoches=20, max_to_keep=5, log_dir="log", checkpoint_dir="checkpoint"):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 创建相关文件夹
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)


            # ===========================================log================================================
            time_str = str(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
            print("log writing to {}\n".format(log_dir))

            # train_log
            if not os.path.exists(os.path.join(log_dir, "train", time_str)):
                os.makedirs(os.path.join(log_dir, "train", time_str))
            train_summary_dir = os.path.join(log_dir, "train", time_str)
            train_summary_op = tf.summary.merge([self.loss_summary, self.acc_summary, self.grad_summaries_merged])
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # dev_log
            if not os.path.exists(os.path.join(log_dir, "dev", time_str)):
                os.makedirs(os.path.join(log_dir, "dev", time_str))
            dev_summary_dir = os.path.join(log_dir, "dev", time_str)
            dev_summary_op = tf.summary.merge([self.loss_summary, self.acc_summary])
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # ===========================================saver=============================================
            checkpoint_dir = os.path.abspath(os.path.join(checkpoint_dir, time_str))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_to_keep)

            def train_step(x_batch, y_batch):
                feed_dict = {
                              self.input_x: x_batch,
                              self.input_y: y_batch,
                              self.max_sentence_num: 30,
                              self.max_sentence_length: 30,
                              }

                _, summaries, loss, accuracy = sess.run([self.train_op, train_summary_op, self.loss, self.acc], feed_dict=feed_dict)
                train_summary_writer.add_summary(summaries)
                return loss, accuracy

            def dev_step(x_batch, y_batch, writer=None):
                feed_dict = {
                              self.input_x: x_batch,
                              self.input_y: y_batch,
                              self.max_sentence_num: 30,
                              self.max_sentence_length: 30,
                              }

                summaries, loss, accuracy = sess.run([dev_summary_op, self.loss, self.acc], feed_dict)

                if writer:
                    writer.add_summary(summaries)
                return accuracy

            for epoch in range(epoches):
                print('epoch: {}'.format(epoch + 1))
                for batch in range(len(train_x)//self.batch_size):
                    #=============================================训练=========================================
                    x = train_x[batch * self.batch_size: (batch + 1) * self.batch_size]
                    y = train_y[batch * self.batch_size: (batch + 1) * self.batch_size]

                    loss, accuracy = train_step(x, y)
                    time_str = str(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
                    print("{}:  epoch: {}/{}  batch:  {}/{}  loss {:g}, acc {:g}".format(time_str, epoch, epoches, batch, len(train_x)//self.batch_size, loss, accuracy))

                #==========================================每个epoch验证一次准确率==============================
                print("Validation......")
                accuracy = []
                for batch in range(len(dev_x)//self.batch_size):
                    x = dev_x[batch * self.batch_size: (batch + 1) * self.batch_size]
                    y = dev_y[batch * self.batch_size: (batch + 1) * self.batch_size]
                    accuracy.append(dev_step(x, y, dev_summary_writer))
                print("========================================================================")
                print("validation_accuracy of epoch {}: {}".format(epoch, np.mean(accuracy)))
                print("========================================================================")




    def BidirectionalGRUEncoder(self, inputs, name):
        #输入inputs的shape是[batch_size, max_time, voc_size]
        with tf.variable_scope(name):
            GRU_cell_fw = rnn.GRUCell(self.hidden_size)
            GRU_cell_bw = rnn.GRUCell(self.hidden_size)
            #fw_outputs和bw_outputs的size都是[batch_size, max_time, hidden_size]
            ((fw_outputs, bw_outputs), (_, _)) = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                                                 cell_bw=GRU_cell_bw,
                                                                                 inputs=inputs,
                                                                                 sequence_length=length(inputs),
                                                                                 dtype=tf.float32)
            #outputs的size是[batch_size, max_time, hidden_size*2]
            outputs = tf.concat((fw_outputs, bw_outputs), 2)
            return outputs

    def AttentionLayer(self, inputs, name):
        #inputs是GRU的输出，size是[batch_size, max_time, encoder_size(hidden_size * 2)]
        with tf.variable_scope(name):
            # u_context是上下文的重要性向量，用于区分不同单词/句子对于句子/文档的重要程度,
            # 因为使用双向GRU，所以其长度为 (2×hidden_size,)
            u_context = tf.Variable(tf.truncated_normal([self.hidden_size * 2]), name='u_context')

            #使用一个全连接层编码GRU的输出的到其隐层表示,输出h的size是[batch_size, max_time, hidden_size * 2]
            h = layers.fully_connected(inputs, self.hidden_size * 2, activation_fn=tf.nn.tanh)

            #shape为[batch_size, max_time, 1]
            alpha = tf.nn.softmax(tf.reduce_sum(tf.multiply(h, u_context), axis=2, keep_dims=True), dim=1)
            #reduce_sum之前shape为[batch_szie, max_time, hidden_szie*2]，之后shape为[batch_size, hidden_size*2]
            atten_output = tf.reduce_sum(tf.multiply(inputs, alpha), axis=1)
            return atten_output
