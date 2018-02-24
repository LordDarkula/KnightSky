import os

from chess_py import *
from KnightSky.models.cnn import train
from KnightSky.preprocessing.arraybuilder import ArrayBuilder
from KnightSky.preprocessing.helpers import featurehelper

if __name__ == '__main__':
    datapath = os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir, 'data'))

    processor = ArrayBuilder(datapath)
    processor.process_files()
    processor.convert_to_arrays()

    evaluator = train.BoardEvaluator(os.path.abspath(os.path.join(os.pardir, os.pardir, os.pardir, 'tmp')))
    evaluator.train_on(location=datapath,
                       epochs=5000,
                       learning_rate=0.00001)

    print("Board evaluation is {}"
          .format(evaluator.eval([featurehelper.extract_features_from_positions(Board.init_default())])))
