# -*- coding: utf-8 -*-

"""
Constructs convolutional neural network
"""

import os
import numpy as np

from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.models import load_model

from definitions import ROOT_DIR
from KnightSky.helpers import oshelper
from KnightSky.preprocessing import split


class BoardEvaluator:
    """
    Wrapper class for this tensorflow model. Creates computational graph
    """
    BOARD_SIZE = 64
    LENGTH = 8
    NUMBER_OF_CLASSES = 3
    SAVE_PATH = os.path.join(ROOT_DIR, 'tmp', 'model')

    def __init__(self, model=None):
        if model is None:
            self.model = Sequential([
                Reshape(input_shape=(64,), target_shape=(8, 8, 1)),
                Conv2D(16, (4, 4), activation='relu'),
                Conv2D(4, (4, 4)),
                Flatten(),
                Dense(16, activation='relu'),
                Dense(3, activation='softmax'),
            ])

            self.model.compile(optimizer=Adam(),
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])
        else:
            self.model = model

    def fit(self, game_features, game_labels):
        positions = []
        advantages = []
        for index, game in enumerate(game_features):
            for position in game:
                positions.append(position)
            for advantage in game_labels[index]:
                advantages.append(advantage)

        self.model.fit(positions, advantages, batch_size=100, epochs=5)

    def save_to_h5(self):
        self.model.save(self.SAVE_PATH)

    @classmethod
    def load_from_h5(cls):
        try:
            return cls(load_model(cls.SAVE_PATH))
        except ValueError:
            raise ValueError("No model found at {}".format(cls.SAVE_PATH))
        except ImportError:
            raise ImportError("h5py not found. Please install dependency (eg. pip install h5py).")


if __name__ == '__main__':
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
    tmp_folder_path = os.path.join(ROOT_DIR, 'tmp')
    data_path = os.path.join(ROOT_DIR, 'data')

    features = np.load(os.path.join(data_path, 'arrays', 'features-combined.npy'))
    labels = np.load(oshelper.pathjoin(data_path, 'arrays', 'labels-combined.npy'))
    print(np.sum(labels == 0))
    print(np.sum(labels == 1))
    print(np.sum(labels == 2))
    labels = to_categorical(labels, num_classes=3)
    wc = 0
    dc = 0
    bc = 0
    # for label in labels:
    #     if label[0] == 1:
    #         print('white')
    #         wc += 1
    #     elif label[1] == 1:
    #         print('draw')
    #         dc += 1
    #     elif label[2] == 1:
    #         print('black')
    #         bc += 1

    print("White wins {} draw {} black wins {}".format(wc, dc, bc))
    train_features, test_features, train_labels, test_labels = split.randomly_assign_train_test(features, labels)

    evaluator = BoardEvaluator()
    evaluator.model.fit(train_features, train_labels, batch_size=32, epochs=500)
    print(evaluator.model.evaluate(test_features, test_labels))
    print(np.array(test_labels[2]))
    print(evaluator.model.predict(np.array(test_features)))
