# -*- coding: utf-8 -*-

"""
Constructs different neural network layers
"""
import tensorflow as tf


def conv_layer(X, W, b, name='conv'):
    with tf.name_scope(name):
        convolution = tf.nn.conv2d(X, W, strides=[1, 1, 1, 1], padding='SAME')
        activation = tf.nn.relu(convolution + b)

        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activation', activation)

        return tf.nn.max_pool(activation, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


def relu_layer(X, W, b, name='relu_layer'):
    with tf.name_scope(name):
        activation = tf.nn.relu(tf.matmul(X, W) + b)

        tf.summary.histogram('weights', W)
        tf.summary.histogram('biases', b)
        tf.summary.histogram('activation', activation)

        return activation


def sigmoid_layer(X, weights, biases, name='sigmoid_layer'):
    with tf.name_scope(name):
        activation = tf.sigmoid(tf.matmul(X, weights) + biases)

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('activation', activation)

        return activation


def tanh_layer(X, weights, biases, name='tanh_layer'):
    with tf.name_scope(name):
        activation = tf.tanh(tf.matmul(X, weights) + biases)

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('biases', biases)
        tf.summary.histogram('activation', activation)

        return activation
