# -*- coding: utf-8 -*-

"""
Class to access all pre-processing functionality.
"""

import os
import numpy as np
import re
import itertools
import json
from chess_py import *

from KnightSky.helpers import oshelper
from KnightSky.preprocessing.helpers import featurehelper


class ArrayBuilder:
    def __init__(self, datapath):
        """
        Creates object that processes data and converts it to numpy arrays.
        Requires original PGN files to be from FICS database and in ``data/raw``.
        Saves processed PGN in ``data/processed`` and np arrays in ``data/arrays``.
        
        :param: datapath: path to data folder
        :type: datapath: str
        """
        if not os.path.exists(datapath):
            raise FileNotFoundError("Please create /raw path and put chess data in there")

        self.paths_dict = {'data': datapath,
                           'raw': oshelper.pathjoin(datapath, "raw"),
                           'processed': oshelper.pathjoin(datapath, "processed", "processed.json"),
                           'arrays': oshelper.pathjoin(datapath, "arrays")}
        self.games = {'games': [],
                      'length': 0}

        oshelper.create_if_not_exists(self.paths_dict['processed'], is_file=True)
        oshelper.create_if_not_exists(self.paths_dict['arrays'], is_file=False)

    def process_files(self):
        """
        Recursive function that goes through all directories in ``data/raw``
        and processes them. Saves output as json in ``data/processed``
        """
        def process_directory(path):
            if os.path.isfile(path):
                self._remove_metadata(path)
            else:
                for group in os.listdir(path):
                    process_directory(os.path.join(path, group))

        process_directory(self.paths_dict['raw'])

        with open(self.paths_dict['processed'], 'w') as f:
            json.dump(self.games, f)
        self.games = {'games': [],
                      'length': 0}

    def _remove_metadata(self, rawpath):
        forfeit = "forfeits by disconnection"
        with open(rawpath, 'r') as raw:

            for line in raw:
                if line[:2] == "1." and forfeit not in line:  # Valid Game
                    end = line.index('{')
                    processed_line = line[:end].replace('+', '').replace('#', '')
                    processed_line = re.sub(r'[1-9][0-9]*\.\s', '', processed_line)
                    result = float(0 if '1-0' in line[end:]
                                   else 1 if '0-1' in line[end:]
                                   else 0.5)

                    processed_game = {'result': result,
                                 'moves': processed_line.strip().split(' ')}
                    self.games['games'].append(processed_game)
                    self.games['length'] += 1

    def convert_to_arrays(self, label_type='material'):
        """
        Converts to two arrays, X and y.

        X is the list of all chess positions in the form
        [number of games, 64 (number of squares on the board)]

        Y is the list of evaluations in the form
        [number of games, 3 (good for white, draw, good for black)]

        Saves output in ``data/arrays`` and returns it.

        :return: features, labels
        :rtype: tuple(np.array, np.array)
        """
        color_dict = {color.white: 0, color.black: 1}

        with open(self.paths_dict['processed'], 'r') as f:
            self.games = json.load(f)
            features = []

            # resets every game
            game_increment = 0

            for game_dict in self.games['games']:
                data_board = Board.init_default()

                for move in game_dict['moves']:

                    try:
                        if game_increment % 2 == 0:
                            current_color = color.white
                        else:
                            current_color = color.black
                        move = converter.incomplete_alg(move, current_color)
                        move = converter.make_legal(move, data_board)
                        data_board.update(move)
                    except Exception as error:
                        print(error)
                        break

                    features.append(featurehelper.extract_features_from_position(data_board))
                    game_increment += 1

                game_increment = 0



        # np.save(oshelper.pathjoin(self.paths_dict['arrays'], 'features'), np.array(features))
        # np.save(oshelper.pathjoin(self.paths_dict['arrays'], 'labels'), np.array(labels))
        #
        # return features, labels


if __name__ == '__main__':
    builder = ArrayBuilder(os.path.abspath(os.path.join(os.pardir, 'data')))
    # builder.process_files()
    builder.convert_to_arrays()
