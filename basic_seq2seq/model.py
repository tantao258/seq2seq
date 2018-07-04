import tensorflow as tf
import tensorflow.contrib.seq2seq as tcs
import tensorflow.contrib.layers as tcl

class basic_seq2seq(object):
    def __init__(self, batch_size,
                 learning_rate,
                 data,
                 n_layers=3,
                 lstm_size=128,
                 encoder_embedding_size=256,
                 decoder_embedding_size=256
                 ):

        self.data = data
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.encoder_embedding_size = encoder_embedding_size
        self.decoder_embedding_size = decoder_embedding_size
        self.n_layers = n_layers
        self.keep_prob = 1
        self.lstm_size = lstm_size
        self.sess = tf.Session()

        self.build_input()
        self.build_embedding()
        self.build_encoder()
        self.build_decoder()
        self.build_output()
        self.build_loss()
        self.build_optimizer()
        self.build_log()
        self.saver = tf.train.Saver(max_to_keep=2)

    def build_input(self):
        with tf.variable_scope("Inputs"):
            self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name="encoder_input")
            self.decoder_target = tf.placeholder(tf.int32, [None, None], name="decoder_target")
            self.decoder_input = tf.placeholder(tf.int32, [None, None], name="decoder_input")
            self.encoder_input_sequence_length = tf.placeholder(tf.int32, (None,), name='encoder_inputs_sequence_length')
            self.decoder_target_sequence_length = tf.placeholder(tf.int32, (None,), name='decoder_target_sequence_length')
            self.decoder_max_target_sequence_length = tf.reduce_max(self.decoder_target_sequence_length, name='max_target_len')

    def build_embedding(self):
        with tf.variable_scope("Embedding"):
            self.encoder_input_embedding = tcl.embed_sequence(self.encoder_inputs,          # [None, None, 15]
                                                              self.data.encoder_vocab_size,
                                                              self.encoder_embedding_size,
                                                              scope="encoder_input_embedding")

            self.decoder_embedding = tf.Variable(tf.random_uniform([self.data.decoder_vocab_size, self.decoder_embedding_size])) # [31,15]
            self.decoder_input_embedding = tf.nn.embedding_lookup(self.decoder_embedding,
                                                                  self.decoder_input,
                                                                  name="decoder_target_embedding") # [None, None, 15]

    def build_encoder(self):
        with tf.variable_scope("Encoder"):
            with tf.name_scope("encoder_cell"):
                # 创建单个cell
                def get_a_cell(lstm_size, keep_prob):
                    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
                    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
                    return drop
                # 堆叠多层神经元
                cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.n_layers)])
            # 初始化神经元状态
            with tf.name_scope("encoder_initial_state"):
                initial_state = cell.zero_state(self.batch_size, tf.float32)
            # 动态展开
            with tf.name_scope("encoder_output"):
                self.encoder_lstm_outputs, self.encoder_final_state = tf.nn.dynamic_rnn(cell=cell,
                                                                                        inputs=self.encoder_input_embedding,
                                                                                        sequence_length=self.encoder_input_sequence_length,
                                                                                        dtype=tf.float32)
            """
            self.encoder_lstm_outputs.shape = [128, ?, 50]
            self.encoder_final_state:
                                     c1.shape = [128, 50]     h1.shape = [128, 50]
                                     c2.shape = [128, 50]     h2.shape = [128,50]
            """

    def build_decoder(self):
        with tf.variable_scope("Decoder"):
            with tf.name_scope("Decoder_cell"):
                # 创建单个cell
                def get_a_cell(lstm_size, keep_prob):
                    lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
                    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
                    return drop
                # 堆叠多层神经元
                cell = tf.nn.rnn_cell.MultiRNNCell([get_a_cell(self.lstm_size, self.keep_prob) for _ in range(self.n_layers)])

            with tf.name_scope("Decoder_Dense"):
                #Output全连接层
                output_layer = tf.layers.Dense(units=self.data.decoder_vocab_size,
                                               kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

            with tf.variable_scope("decoder"):
                # 创建helper对象
                training_helper = tcs.TrainingHelper(inputs=self.decoder_input_embedding,
                                                     sequence_length=self.decoder_target_sequence_length,
                                                     time_major=False)
                # 构造decoder
                training_decoder = tcs.BasicDecoder(cell=cell,
                                                    helper=training_helper,
                                                    initial_state=self.encoder_final_state,
                                                    output_layer=output_layer)

                self.training_decoder_output, \
                self.training_decoder_final_state, \
                self.training_decoder_final_sequence_lengths = tcs.dynamic_decode(decoder=training_decoder,
                                                                                  output_time_major=False,
                                                                                  impute_finished=True,
                                                                                  maximum_iterations=self.decoder_max_target_sequence_length)
            # prediction
            with tf.variable_scope("decoder", reuse=True):
                # 创建一个常量tensor并复制为batch_size的大小
                start_tokens = tf.tile(tf.constant([self.data.decoder_word_to_int['<GO>']], dtype=tf.int32),
                                       [self.batch_size],
                                       name='start_tokens')

                predicting_helper = tcs.GreedyEmbeddingHelper(embedding=self.decoder_embedding,
                                                              start_tokens=start_tokens,
                                                              end_token=self.data.decoder_word_to_int['<EOS>'])

                predicting_decoder = tcs.BasicDecoder(cell=cell,
                                                      helper=predicting_helper,
                                                      initial_state=self.encoder_final_state,
                                                      output_layer=output_layer)

                self.predicting_decoder_output, \
                self.predicting_decoder_final_state, \
                self.predicting_decoder_final_sequence_lengths = tcs.dynamic_decode(decoder=predicting_decoder,
                                                                                    output_time_major=False,
                                                                                    impute_finished=True,
                                                                                    maximum_iterations=self.decoder_max_target_sequence_length)

    def build_output(self):
        """
            tf.contrib.seq2seq.dynamic_decode 用于构造一个动态的decoder，返回的内容是：
            (final_outputs, final_state, final_sequence_lengths).
            其中，final_outputs是一个named tuple，里面包含两项(rnn_outputs, sample_id)
            rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
            sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
        """
        self.training_logits = tf.identity(self.training_decoder_output.rnn_output, 'logits')
        # self.training_decoder_output.rnn_output.shape = [128, None, 31]
        self.predicting_logits = tf.identity(self.predicting_decoder_output.sample_id, name='predictions')

    def build_loss(self):
        with tf.variable_scope("Loss"):
            masks = tf.sequence_mask(lengths=self.decoder_target_sequence_length,
                                     maxlen=self.decoder_max_target_sequence_length,
                                     dtype=tf.float32,
                                     name='masks')

            self.loss = tcs.sequence_loss(logits=self.training_logits,
                                          targets=self.decoder_target,
                                          weights=masks)
            tf.summary.scalar("loss", self.loss)

    def build_optimizer(self):
        with tf.variable_scope("Optimizer"):
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.loss)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)

    def build_log(self):
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter("log/", self.sess.graph)

    def train(self, epoches=60):
        self.sess.run(tf.global_variables_initializer())

        # =====================================训练==============================================
        for epoch in range(epoches+1):
            for batch in range(len(self.data.encoder_int)//self.batch_size):
                pad_encoder_input_batch, \
                pad_decoder_input_batch, \
                pad_decoder_target_batch, \
                encoder_input_length, \
                decoder_target_length = self.data.get_train_batch(self.batch_size, batch)

                train_feed_dict = {
                                    self.encoder_inputs: pad_encoder_input_batch,
                                    self.encoder_input_sequence_length: encoder_input_length,
                                    self.decoder_input: pad_decoder_input_batch,
                                    self.decoder_target: pad_decoder_target_batch,
                                    self.decoder_target_sequence_length: decoder_target_length,
                                    }

                summary, _, training_loss = self.sess.run([self.merged, self.train_op, self.loss], feed_dict=train_feed_dict)
                self.writer.add_summary(summary)

                print("epoch: {}/{}    batch: {}/{}    loss: {}".format(epoch, epoches, batch, len(self.data.encoder_int)//self.batch_size, training_loss))

            # 保存模型
            if epoch % 10 == 0 and epoch != 0:
                self.saver.save(self.sess, "./checkpoint/model.ckpt", global_step=epoch)

    def prediction(self):
        def source_to_seq(text):
            """
            对源数据进行转换
            """
            sequence_length = 7
            return [self.data.encoder_word_to_int.get(word, self.data.encoder_word_to_int["<UNK>"]) for word in text] +\
                   [self.data.encoder_word_to_int["<PAD>"]]*(sequence_length - len(text))

        self.sess.run(tf.global_variables_initializer())
        # 加载模型
        model_path = tf.train.latest_checkpoint("./checkpoint/")
        print('Restored from: {}'.format(model_path))
        self.saver.restore(self.sess, model_path)

        # 输入
        input_word = "wqzsygxb"
        text = source_to_seq(input_word)

        prediction = self.sess.run(self.predicting_logits, feed_dict={self.encoder_inputs: [text] * self.batch_size,
                                                                      self.encoder_input_sequence_length: [len(input_word)] * self.batch_size,
                                                                      self.decoder_target_sequence_length: [len(input_word) * self.batch_size]
                                                                      })


        print("原始输入：", input_word)
        print("\nInput")
        print("  word 编号：   {}".format([i for i in text]))
        print("  Input Words:  {}".format(" ".join([self.data.encoder_int_to_word[i] for i in text])))

        print("\nTarget")
        print("  Word 编号:    {}".format([i for i in prediction[0] if i != self.data.encoder_word_to_int["<PAD>"]]))
        print("  Response Words: {}".format(" ".join([self.data.encoder_int_to_word[i] for i in prediction[0] if i != self.data.encoder_word_to_int["<PAD>"]])))