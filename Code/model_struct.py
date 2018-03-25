'''
Created by Guy Tordjman (turgibot@gmail.com)
March 25th 2018
Please always use your code for the benefit of humanity.
'''

import tensorflow as tf


class Model:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self._predicition = None
        self._optimize = None
        self._error = None


    @property
    def predicition(self):
        if not self._predicition:
            data_size = int(self.data.get_shape()[1])
            target_size = int(self.target.get_shape()[1])
            weight = tf.Variable(tf.truncated_normal([data_size, target_size]))
            bias = tf.Variable(tf.constant(0.1, shape=[target_size]))
            incoming = tf.matmul(self.data, weight) + bias
            self._predicition = tf.nn.softmax(incoming)
        return self._predicition

    @property
    def optimize(self):
        if not self._optimize:
            cross_entrpy = -tf.reduce_sum(self.target, tf.log(self.predicition))
            optimizer = tf.train.RMSPropOptimizer(0.03)
            self._optimize = optimizer.minimize(cross_entrpy)
        return self._optimize

    @property
    def error(self):
        if not self._error:
            mistakes = tf.not_equal(tf.argmax(self.target, 1), tf.argmax(self.predicition, 1))
            self._error = tf.reduce_mean(tf.cast(mistakes, tf.float32))
        return self._error
