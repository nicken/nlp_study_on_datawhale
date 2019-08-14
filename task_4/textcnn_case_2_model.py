# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


class Settings(object):
    """
    configuration class
    """

    def __init__(self, vocab_size=100000, embedding_size=128):
        self.model_name = "CNN"
        self.embedding_size = embedding_size
        self.filter_size = [2, 3, 4, 5]
        self.n_filters = 128
        self.fc_hidden_size = 1024
        self.n_class = 2
        self.vocab_size = vocab_size
        self.max_words_in_doc = 20


class TextCNN(object):
    """
    Text CNN
    """

    def __init__(self, settings, pre_trained_word_vectors=None):
        self.model_name = settings.model_name
        self.embedding_size = settings.embedding_size
        self.filter_size = settings.filter_size
        self.n_filter = settings.n_filters
        self.fc_hidden_size = settings.fc_hidden_size
        self.n_filter_total = self.n_filter * (len(self.filter_size))
        self.n_class = settings.n_class
        self.max_words_in_doc = settings.max_words_in_doc
        self.vocab_size = settings.vocab_size

        """ 定义网络的结构 """
        # 输入样本
        with tf.name_scope("inputs"):
            self._inputs_x = tf.placeholder(tf.int64, [None, self.max_words_in_doc], name="_inputs_x")
            self._inputs_y = tf.placeholder(tf.float16, [None, self.n_class], name="_inputs_y")
            self._keep_dropout_prob = tf.placeholder(tf.float32, name="_keep_dropout_prob")

        # 嵌入层
        with tf.variable_scope("embedding"):
            if isinstance(pre_trained_word_vectors, np.ndarray):  # 使用预训练的词向量
                assert isinstance(pre_trained_word_vectors,
                                  np.ndarray), "pre_trained_word_vectors must be a numpy's ndarray"
                assert pre_trained_word_vectors.shape[
                           1] == self.embedding_size, "number of col of pre_trained_word_vectors must euqals embedding size"
                self.embedding = tf.get_variable(name='embedding',
                                                 shape=pre_trained_word_vectors.shape,
                                                 initializer=tf.constant_initializer(pre_trained_word_vectors),
                                                 trainable=True)
            else:
                self.embedding = tf.Variable(tf.truncated_normal((self.vocab_size, self.embedding_size)))

        # conv-pool
        inputs = tf.nn.embedding_lookup(self.embedding,
                                        self._inputs_x)  # [batch_size, words, embedding]  # look up layer
        inputs = tf.expand_dims(inputs, -1)  # [batch_size, words, embedding, 1]
        pooled_output = []

        for i, filter_size in enumerate(self.filter_size):  # filter_size = [2, 3, 4, 5]
            with tf.variable_scope("conv-maxpool-%s" % filter_size):
                # conv layer
                filter_shape = [filter_size, self.embedding_size, 1, self.n_filter]
                W = self.weight_variable(shape=filter_shape, name="W_filter")
                b = self.bias_variable(shape=[self.n_filter], name="b_filter")
                conv = tf.nn.conv2d(inputs, W, strides=[1, 1, 1, 1], padding="VALID",
                                    name='text_conv')  # [batch, words-filter_size+1, 1, channel]
                # apply activation
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # max pooling
                pooled = tf.nn.max_pool(h, ksize=[1, self.max_words_in_doc - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1], padding="VALID",
                                        name='max_pool')  # [batch, 1, 1, channel]
                pooled_output.append(pooled)

        h_pool = tf.concat(pooled_output, 3)  # concat on 4th dimension
        self.h_pool_flat = tf.reshape(h_pool, [-1, self.n_filter_total], name="h_pool_flat")

        # add dropout
        with tf.name_scope("dropout"):
            self.h_dropout = tf.nn.dropout(self.h_pool_flat, self._keep_dropout_prob, name="dropout")

        # output layer
        with tf.name_scope("output"):
            W = self.weight_variable(shape=[self.n_filter_total, self.n_class], name="W_out")
            b = self.bias_variable(shape=[self.n_class], name="bias_out")
            self.scores = tf.nn.xw_plus_b(self.h_dropout, W, b, name="scores")  # class socre
            print("self.scores : ", self.scores.get_shape())
            self.predictions = tf.argmax(self.scores, 1, name="predictions")  # predict label , the output
            print("self.predictions : ", self.predictions.get_shape())

    # 辅助函数
    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)
