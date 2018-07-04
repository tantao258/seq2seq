#coding=utf-8
import tensorflow as tf
import time
import os
from utils import *
from HAN_model import *


# Data loading params
tf.flags.DEFINE_string("yelp_json_path", 'data/yelp_academic_dataset_review.json', "data directory")
tf.flags.DEFINE_integer("vocab_size", 46960, "vocabulary size")
tf.flags.DEFINE_integer("num_classes", 5, "number of classes")
tf.flags.DEFINE_integer("embedding_size", 200, "Dimensionality of character embedding (default: 200)")
tf.flags.DEFINE_integer("hidden_size", 50, "Dimensionality of GRU hidden layer (default: 50)")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 50)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_sent_in_doc", 30, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("max_word_in_sent", 30, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_integer("evaluate_every", 100, "evaluate every this many batches")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_float("grad_clip", 5, "grad clip to prevent gradient explode")
FLAGS = tf.flags.FLAGS


# 数据加载
data = DataProcess()
train_x, train_y, dev_x, dev_y = data.data_load()
print("data load finished")


# 创建HAN对象
han = HAN(learning_rate=FLAGS.learning_rate,
          batch_size=FLAGS.batch_size,
          grad_clip=FLAGS.grad_clip,
          vocab_size=FLAGS.vocab_size,
          num_classes=FLAGS.num_classes,
          embedding_size=FLAGS.embedding_size,
          hidden_size=FLAGS.hidden_size
          )

han.train(train_x=train_x,
          train_y=train_y,
          dev_x=dev_x,
          dev_y=dev_y,
          epoches=FLAGS.num_epochs)