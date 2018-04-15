# -*- coding: utf-8 -*-

"""
Constructs convolutional neural network
"""

import os
import numpy as np
from chess_py import *

from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape
from keras.layers import ConvLSTM2D, LSTM, Conv2D
from keras.optimizers import Adam
from keras.models import load_model

from definitions import ROOT_DIR
from KnightSky.helpers import oshelper
from KnightSky.preprocessing.helpers import featurehelper


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
                LSTM(3, return_sequences=True, input_shape=(332, 64), use_bias=True, activation='softmax'),
            ])
            # self.model = Sequential([
            #     Reshape(input_shape=(332, 64), target_shape=(332, 8, 8, 1)),  # Channels last
            #     ConvLSTM2D(64, (4, 4), padding='valid', return_sequences=True),
            #     ConvLSTM2D(3, (2, 2), padding='valid', return_sequences=True),
            #     Reshape(target_shape=(332, 3)),
            # ])

            self.model.compile(optimizer='rmsprop',
                               loss='mean_squared_error',
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

    features = np.load(os.path.join(data_path, 'arrays', 'features-split.npy'))
    labels = np.load(oshelper.pathjoin(data_path, 'arrays', 'labels-split.npy'))
    labels = [to_categorical(l, num_classes=3) for l in labels]

    piece_id = piece_const.PieceValues.init_manual(pawn_value=1,
                                                   knight_value=2,
                                                   bishop_value=3,
                                                   rook_value=4,
                                                   queen_value=5,
                                                   king_value=6)
    padding_board = featurehelper.extract_features_from_position(Board.init_default(), piece_id)
    features = sequence.pad_sequences(features, padding='pre', value=padding_board)
    labels = sequence.pad_sequences(labels, padding='pre', value=[0, 1, 0])

    print(features[1].shape)

    feature_index = int(0.9*len(features))
    label_index = int(0.9*len(labels))
    train_features, test_features, train_labels, test_labels = \
        features[:feature_index], features[feature_index:], \
        labels[:label_index], labels[label_index:]
    print("Train {} {} {} Test {} {} {}".format(np.sum(train_labels == [1, 0, 0]),
                                                np.sum(train_labels == [0, 1, 0]),
                                                np.sum(train_labels == [1, 0, 0]),
                                                np.sum(test_labels == [1, 0, 0]),
                                                np.sum(test_labels == [0, 1, 0]),
                                                np.sum(test_labels == [1, 0, 0]),
                                                ))

    evaluator = BoardEvaluator()
    evaluator.model.fit(train_features, train_labels, batch_size=10, epochs=2)
    print(evaluator.model.evaluate(test_features, test_labels))
    print(np.array(test_labels[2]))
    print(evaluator.model.predict(test_features).shape)
