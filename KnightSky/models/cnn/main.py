import os

from chess_py import *
from KnightSky.models.cnn import train
from KnightSky.preprocessing.helpers import featurehelper

if __name__ == '__main__':
    evaluator = train.BoardEvaluator(os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir, 'tmp')))
    evaluator.train_on(os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir, 'data')),
                       epochs=5000,
                       learning_rate=0.00001)
    evaluator.eval([featurehelper.extract_features(Board.init_default())])
