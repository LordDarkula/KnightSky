# -*- coding: utf-8 -*-

"""
Constructs convolutional neural network
"""

import os
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

from definitions import ROOT_DIR
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

    def __init__(self):
        self.model = Sequential([
            Dense(32, input_dim=64),
            Activation('relu'),
            Dense(3),
            Activation('softmax')
        ])
        self.model.compile(optimizer=Adam(),
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    @classmethod
    def from_saved(cls):
        pass

    def _create_model(self):
        model = Sequential([
            Dense(32, input_shape=(None, 64)),
            Activation('relu'),
            Dense(3),
            Activation('softmax')
        ])
        model.compile(optimizer=Adam(),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])



    def fit(self,
            training_data,
            testing_data,
            epochs=300,
            batch_size=100,
            learning_rate=0.5,
            train_keep_prob=0.5,
            test_keep_prob=1.0):
        """
        Trains on data in the form of a tuple of np arrays stored as (X, y),
        or the path to a folder containing 2 npy files 
        wih X named ``features.npy`` and y named ``labels.npy``.
        
        :param training_data: data used to train the network
        :type: Tuple[np.array, np.array]
        :param testing_data: data used to test the neural network
        :param epochs: Number of epochs to run when training. Defaults to 300.
        :param batch_size: Size of individual batches to 
        :param learning_rate: 
        :param train_keep_prob: 
        :param test_keep_prob: 
        """

        positions, advantages = training_data
        test_positions, test_advantages = testing_data
        saver = tf.train.Saver()

        # sess_config = tf.ConfigProto(log_device_placement=True)
        # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.5
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
                        enumerate(split.next_batch(positions, advantages, batch_size=batch_size)):

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

                accuracy_dict = {self.X_placeholder:         test_positions,
                                 self.y_placeholder:         test_advantages,
                                 self.keep_prob_placeholder: test_keep_prob}

                print("Test accuracy {}".format(self.accuracy.eval(accuracy_dict)))

            # tf.add_to_collection('evaluate', self.evaluate)
            saver.save(sess, self.save_path)

    def predict(self, positions):
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.save_path)
            # self.evaluate = tf.get_collection('evaluate')[0]
            advantages = self.predict_op.eval(feed_dict={self.X_placeholder: positions,
                                                         self.keep_prob_placeholder: 1.0})
            print("Position evaluation is {}".format(advantages))
            return advantages


if __name__ == '__main__':
    tmp_folder_path = os.path.join(ROOT_DIR, 'tmp')
    data_path = os.path.join(ROOT_DIR, 'data')

    features = np.load(os.path.join(data_path, 'arrays', 'features.npy'))
    labels = np.load(oshelper.pathjoin(data_path, 'arrays', 'labels.npy'))
    train_features, test_features, train_labels, test_labels = split.randomly_assign_train_test(features, labels)

    evaluator = BoardEvaluator()
    evaluator.model.fit(train_features, train_labels, batch_size=10, epochs=5)
    print(evaluator.model.evaluate(test_features, test_labels))

