import numpy as np
import os

class Data(object):
    def __init__(self):
        self.input_data_path = "./data/letters_source.txt"
        self.target_data_path = "./data/letters_target.txt"

        self.encoder_data()
        self.decoder_data()

    def data_process(self, data_path):
        """
        1、建立词典
        2、将数据转化为类型
        :param data_path: 需要处理数据的路劲
        :return: int_to_word        映射
                  word_to_int        映射
                  len(int_to_word)   词典大小
                  text_int           转化后的文本
        """
        with open(data_path, "r", encoding="utf-8") as f:
            data = f.read()
        special_words = ["<PAD>", "<UNK>", "<GO>", "<EOS>"]
        set_words = list(set([character for line in data.split('\n') for character in line]))

        int_to_word = {i: word for i, word in enumerate(special_words + set_words)}
        word_to_int = {word: i for i, word in int_to_word.items()}

        text_int = [[word_to_int.get(letter, word_to_int['<UNK>']) for letter in line] for line in data.split('\n')]
        return int_to_word, word_to_int, len(int_to_word), text_int

    def encoder_data(self):
        self.encoder_int_to_word, self.encoder_word_to_int, \
        self.encoder_vocab_size, self.encoder_int = self.data_process(self.input_data_path)
        self.encoder_input_length = [len(line) for line in self.encoder_int]
        # 求取最长长度
        self.encoder_max_length = max([len(line) for line in self.encoder_int])

    def decoder_data(self):
        self.decoder_int_to_word, self.decoder_word_to_int, \
        self.decoder_vocab_size, self.decoder_int = self.data_process(self.target_data_path)

        # 对 decoder_target 添加终止符<EOS>
        self.decoder_target_int = [item + [self.decoder_word_to_int['<EOS>']] for item in self.decoder_int]
        self.decoder_target_length = [len(line) for line in self.decoder_target_int]

        # 对 decoder_input 添加开始符<GO>
        self.decoder_input_int = [[self.decoder_word_to_int['<GO>']] + item for item in self.decoder_int]
        self.decoder_input_length = [len(line) for line in self.decoder_input_int]

        # 求取最长长度
        self.decoder_max_length = max([len(line) for line in self.decoder_int]) + 1

    def get_train_batch(self, batch_size, batch):
        def pad_sentence_batch(sentence_batch, pad_int):
            """
                对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length
                - sentence batch
                - pad_int: <PAD>对应索引号
            """
            max_length = max([len(sentence) for sentence in sentence_batch])
            return [sentence + [pad_int]*(max_length - len(sentence)) for sentence in sentence_batch]

        encoder_input_batch = self.encoder_int[batch * batch_size:(batch+1) * batch_size]
        decoder_input_batch = self.decoder_input_int[batch * batch_size:(batch+1) * batch_size]
        decoder_target_batch = self.decoder_target_int[batch * batch_size:(batch+1) * batch_size]

        pad_encoder_input_batch = pad_sentence_batch(encoder_input_batch, self.encoder_word_to_int["<PAD>"])
        pad_decoder_target_batch = pad_sentence_batch(decoder_target_batch, self.decoder_word_to_int["<PAD>"])
        pad_decoder_input_batch = pad_sentence_batch(decoder_input_batch, self.decoder_word_to_int["<PAD>"])

        # 记录数据 pad 之前的长度
        encoder_input_length = [len(sentence) for sentence in encoder_input_batch]
        decoder_target_length = [len(sentence) for sentence in decoder_target_batch]

        return pad_encoder_input_batch, pad_decoder_input_batch, pad_decoder_target_batch, encoder_input_length, decoder_target_length

    def get_validation(self, batch_size):
            validation_encoder_input = self.encoder_int[:batch_size]
            validation_decoder_target = self.decoder_target_int[:batch_size]

            return validation_encoder_input, validation_decoder_target

