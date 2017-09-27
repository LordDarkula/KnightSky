# -*- coding: utf-8 -*-

"""
Constructs convolutional neural network
"""

import numpy as np
import tensorflow as tf

from KnightSky.helpers import oshelper
from KnightSky.models.cnn.helpers import tensorboardsetup
from KnightSky.models.cnn.helpers import layers
from KnightSky.models.cnn.helpers.variables import weight_variable, bias_variable
from KnightSky.preprocessing.split import randomly_assign_train_test, next_batch


class BoardEvaluator:
    """
    Wrapper class for this tensorflow model. Creates computational graph
    """
    BOARD_SIZE = 64
    LENGTH = 8
    NUMBER_OF_CLASSES = 3

    def __init__(self, tmp_path):
        # Tensorboard Setup
        self.tmp_path = tmp_path
        oshelper.create_if_not_exists(tmp_path)
        self.tb_dir = tensorboardsetup.current_run_directory(tmp_path)

        # Placholder initialization
        self.X_placeholder = tf.placeholder(tf.float32, [None, self.BOARD_SIZE], name="X")
        self.y_placeholder = tf.placeholder(tf.float32, [None, self.NUMBER_OF_CLASSES], name="y")
        self.keep_prob_placeholder = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32)

        # Model creation
        self.optimizer = None
        self.evaluate = None
        self.accuracy = None
        self._create_model()

    @classmethod
    def from_saved(cls):
        pass

    def _create_model(self):
        X = tf.reshape(self.X_placeholder, [-1, self.LENGTH, self.LENGTH, 1])

        conv1 = {'weights': weight_variable([6, 6, 1, 32]),
                 'biases':  bias_variable([32])}
        model = layers.conv_layer(X, conv1['weights'], conv1['biases'], name='conv1')

        conv2 = {'weights': weight_variable([2, 2, 32, 64]),
                 'biases': bias_variable([64])}
        model = layers.conv_layer(model, conv2['weights'], conv2['biases'], name='conv2')

        model = tf.reshape(model, [-1, 2*2*self.BOARD_SIZE])

        w1 = weight_variable([2*2*self.BOARD_SIZE, 1024])
        b1 = bias_variable([1024])

        model = layers.relu_layer(model, w1, b1, name='fc1')

        model = tf.nn.dropout(model, self.keep_prob_placeholder)

        w_out = weight_variable([1024, 3])
        b_out = bias_variable([3])

        y_predicted = tf.matmul(model, w_out) + b_out

        with tf.name_scope("cross_entropy"):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.y_placeholder, logits=y_predicted))
            tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope("train"):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(cross_entropy)

        self.evaluate = tf.argmax(y_predicted, 1)

        with tf.name_scope("accuracy"):
            correct_prediction = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(self.y_placeholder, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", self.accuracy)

    def train_on(self,
                 data_or_path,
                 array_folder_name='arrays',
                 epochs=300,
                 batch_size=100,
                 learning_rate=0.5,
                 train_keep_prob=0.5,
                 test_keep_prob=1.0):
        """
        Trains on data in the form of a tuple of np arrays stored as (X, y),
        or the path to a folder containing 2 npy files 
        wih X named ``features.npy`` and y named ``labels.npy``.
        
        :param data_or_path: training data
        :type: str or np.array
        :param array_folder_name: name of folder where arrays are stored. Defaults to ``arrays``
        :param epochs: Number of epochs to run when training. Defaults to 300.
        :param batch_size: Size of individual batches to 
        :param learning_rate: 
        :param train_keep_prob: 
        :param test_keep_prob: 
        """
        if isinstance(data_or_path, str):
            features = np.load(oshelper.pathjoin(data_or_path, array_folder_name, 'features.npy'))
            labels = np.load(oshelper.pathjoin(data_or_path, array_folder_name, 'labels.npy'))
        else:
            features, labels = data_or_path

        train_features, test_features, train_labels, test_labels = randomly_assign_train_test(features, labels)

        with tf.Session() as sess:
            print("Session starting")

            # Initialization
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.tb_dir)
            writer.add_graph(sess.graph)
            sess.run(tf.global_variables_initializer())

            # Training loop
            for epoch in range(epochs):
                for i, (batch_X, batch_y) in \
                        enumerate(next_batch(train_features, train_labels, batch_size=batch_size)):

                    # Dict fed to train model
                    train_dict = {self.X_placeholder:         batch_X,
                                  self.y_placeholder:         batch_y,
                                  self.keep_prob_placeholder: train_keep_prob,
                                  self.learning_rate:         learning_rate}

                    sess.run(self.optimizer, feed_dict=train_dict)

                    # Write to tensorboard
                    s = sess.run(merged_summary, feed_dict=train_dict)
                    writer.add_summary(s, i)

                accuracy_dict = {self.X_placeholder:         train_features,
                                 self.y_placeholder:         train_labels,
                                 self.keep_prob_placeholder: train_keep_prob}

                print("Train accuracy {}".format(self.accuracy.eval(accuracy_dict)))

                accuracy_dict[self.keep_prob_placeholder] = test_keep_prob
                print("Test  accuracy {}".format(self.accuracy.eval(accuracy_dict)))
