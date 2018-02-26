# -*- coding: utf-8 -*-

"""
Constructs convolutional neural network
"""

import numpy as np
import tensorflow as tf

from KnightSky.helpers import oshelper
from KnightSky.models.cnn.helpers.tensorboardsetup import TensorboardManager
from KnightSky.models.cnn.helpers import layers
from KnightSky.models.cnn.helpers import variables as var
from KnightSky.preprocessing import split


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
        self.save_path = oshelper.pathjoin(self.tmp_path, 'saved', 'model')
        oshelper.create_if_not_exists(tmp_path, is_file=False)
        self.tb_manager = TensorboardManager(tmp_path)

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

        conv1 = {'weights': var.weight_variable([4, 4, 1, 64]),
                 'biases':  var.bias_variable([64])}
        model = layers.conv_layer(X, conv1['weights'], conv1['biases'], name='conv1')

        conv2 = {'weights': var.weight_variable([2, 2, 64, 64]),
                 'biases':  var.bias_variable([64])}
        model = layers.conv_layer(model, conv2['weights'], conv2['biases'], name='conv2')

        model = tf.reshape(model, [-1, 2*2*self.BOARD_SIZE])

        w1 = var.weight_variable([2*2*self.BOARD_SIZE, 1024])
        b1 = var.bias_variable([1024])

        model = layers.relu_layer(model, w1, b1, name='fc1')

        model = tf.nn.dropout(model, self.keep_prob_placeholder)

        w_out = var.weight_variable([1024, 3])
        b_out = var.bias_variable([3])

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
                 location,
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
        
        :param location: training data
        :type: str or np.array
        :param array_folder_name: name of folder where arrays are stored. Defaults to ``arrays``
        :param epochs: Number of epochs to run when training. Defaults to 300.
        :param batch_size: Size of individual batches to 
        :param learning_rate: 
        :param train_keep_prob: 
        :param test_keep_prob: 
        """
        if isinstance(location, str):
            features = np.load(oshelper.pathjoin(location, array_folder_name, 'features.npy'))
            labels = np.load(oshelper.pathjoin(location, array_folder_name, 'labels.npy'))
        else:
            features, labels = location

        train_features, test_features, train_labels, test_labels = split.randomly_assign_train_test(features, labels)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            print("Session starting")

            # Initialization
            merged_summary = tf.summary.merge_all()
            writer = tf.summary.FileWriter(self.tb_manager.tensorboard_path)
            writer.add_graph(sess.graph)
            self.tb_manager += 1
            sess.run(tf.global_variables_initializer())

            # Training loop
            for epoch in range(epochs):
                for i, (batch_X, batch_y) in \
                        enumerate(split.next_batch(train_features, train_labels, batch_size=batch_size)):

                    # Dict fed to train model
                    train_dict = {self.X_placeholder:         batch_X,
                                  self.y_placeholder:         batch_y,
                                  self.keep_prob_placeholder: train_keep_prob,
                                  self.learning_rate:         learning_rate}

                    sess.run(self.optimizer, feed_dict=train_dict)

                    # Write to tensorboard
                    if i % 5 == 0:
                        s = sess.run(merged_summary, feed_dict=train_dict)
                        writer.add_summary(s, i)

                accuracy_dict = {self.X_placeholder:         train_features,
                                 self.y_placeholder:         train_labels,
                                 self.keep_prob_placeholder: train_keep_prob}

                print("Train accuracy {}".format(self.accuracy.eval(accuracy_dict)))

                accuracy_dict[self.keep_prob_placeholder] = test_keep_prob
                print("Test  accuracy {}".format(self.accuracy.eval(accuracy_dict)))

            saver.save(sess, self.save_path)

    def eval(self, features):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            return float(self.evaluate.eval(feed_dict={self.X_placeholder: features,
                                                       self.keep_prob_placeholder: 1.0},
                                            session=sess))

    def restore(self):
        # tf.reset_default_graph()
        #
        # self.evaluate = tf.get_variable('evaluate', shape=[self.NUMBER_OF_CLASSES])
        #
        # saver = tf.train.Saver()
        #
        # with tf.Session() as sess:
        #     saver.restore(sess, oshelper.pathjoin(self.tmp_path, 'model.ckpt'))
        #     print("Position evaluation is {}".format(self.evaluate.eval(feed_dict={self.X_placeholder: features,
        #                                              self.keep_prob_placeholder: 1.0})))
        pass
