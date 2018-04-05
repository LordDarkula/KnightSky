# -*- coding: utf-8 -*-

"""
Constructs convolutional neural network
"""

import os
import numpy as np

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


if __name__ == '__main__':
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    tmp_folder_path = os.path.join(ROOT_DIR, 'tmp')
    data_path = os.path.join(ROOT_DIR, 'data')

    features = np.load(os.path.join(data_path, 'arrays', 'features-result.npy'))
    labels = np.load(oshelper.pathjoin(data_path, 'arrays', 'labels-result.npy'))
    train_features, test_features, train_labels, test_labels = split.randomly_assign_train_test(features, labels)

    evaluator = BoardEvaluator()
    print(np.array(train_features[0]).shape)
    evaluator.model.fit(np.array(train_features[0]), np.array(train_labels[0]), batch_size=10, epochs=20)
    print(evaluator.model.evaluate(np.array(test_features[0]), np.array(test_labels[0])))
