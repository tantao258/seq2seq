import tensorflow as tf
import numpy as np

batch_size = 100
EOS = 1
def random_sequences(length_from, length_to, vocab_lower, vocab_upper, batch_size):
    def random_length():
        if length_from == length_to:
            return length_from
        return np.random.randint(length_from, length_to + 1)    #返回length_from——length_to的一个随机整数

    while True:
        yield [np.random.randint(low=vocab_lower, high=vocab_upper, size=random_length()).tolist() for _ in range(batch_size)]

batches = random_sequences(length_from=3,
                           length_to=10,
                           vocab_lower=2,
                           vocab_upper=10,
                           batch_size=batch_size)

def make_batch(inputs, max_sequence_length=None):
    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)
    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)   #返回输入的句子的最长长度
    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)
    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element
    inputs_time_major = inputs_batch_major.swapaxes(0, 1)
    return inputs_time_major, sequence_lengths



class seq2seq():
    def __init__(self):
        self.input_embedding_size = 20
        self.encoder_hidden_units = 20
        self.decoder_hidden_units = 20
        self.vocab_size = 10
        self.sess = tf.Session()

        self.build_inputs()
        self.build_embedding()
        self.build_encoder()
        self.build_decoder()
        self.build_output()
        self.build_loss()
        self.build_optimizer()
        self.build_log()

    def build_inputs(self):
        with tf.name_scope("inputs"):
            self.encoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name="encoder_inputs")
            self.decoder_inputs = tf.placeholder(tf.int32, shape=(None, None), name="decoder_inputs")
            self.decoder_targets = tf.placeholder(tf.int32, shape=(None, None), name='decoder_targets')

    def build_embedding(self):
        with tf.name_scope("embedding"):
            embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.input_embedding_size], -1.0, 1.0), dtype=tf.float32)
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.encoder_inputs)
            self.decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, self.decoder_inputs)

    def build_encoder(self):
        encoder_cell = tf.contrib.rnn.LSTMCell(self.encoder_hidden_units)
        encoder_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                                      self.encoder_inputs_embedded,
                                                                      dtype=tf.float32,
                                                                      scope="encoder")

    def build_decoder(self):
        decoder_cell = tf.contrib.rnn.LSTMCell(self.decoder_hidden_units)
        self.decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                                      self.decoder_inputs_embedded,
                                                                      initial_state=self.encoder_final_state,
                                                                      dtype=tf.float32,
                                                                      scope="decoder")

    def build_output(self):
        with tf.variable_scope("output"):
            self.decoder_logits = tf.contrib.layers.linear(self.decoder_outputs, self.vocab_size)
            self.decoder_prediction = tf.argmax(self.decoder_logits, 2)

    def build_loss(self):
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.decoder_targets,
                                                                                                 depth=self.vocab_size,
                                                                                                 dtype=tf.float32),
                                                                               logits=self.decoder_logits))
            tf.summary.scalar("loss", self.loss)

    def build_optimizer(self):
        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

    def build_log(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("log/", self.sess.graph)

    def train(self, epochs=100000):
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            batch = next(batches)
            encoder_inputs_, _ = make_batch(batch)
            decoder_targets_, _ = make_batch([(sequence) + [EOS] for sequence in batch])
            decoder_inputs_, _ = make_batch([[EOS] + (sequence) for sequence in batch])
            feed_dict = {self.encoder_inputs: encoder_inputs_,
                         self.decoder_inputs: decoder_inputs_,
                         self.decoder_targets: decoder_targets_,
                         }

            summary, _, l = self.sess.run([self.merged, self.train_op, self.loss], feed_dict)

            self.writer.add_summary(summary, epoch)

            if epoch == 0 or epoch % 1000 == 0:
                print('loss: {}'.format(self.sess.run(self.loss, feed_dict)))
                predict_ = self.sess.run(self.decoder_prediction, feed_dict)
                for i, (inp, pred) in enumerate(zip(feed_dict[self.encoder_inputs].T, predict_.T)):
                    print('input > {}'.format(inp))
                    print('predicted > {}'.format(pred))
                    if i >= 20:
                        break



#main
s=seq2seq()
s.train()