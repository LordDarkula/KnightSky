# -*- coding: utf-8 -*-

import os
import numpy as np
import re
import itertools
from chess_py import *

from KnightSky.helpers import oshelper
from KnightSky.preprocessing.helpers.featurehelper import extract_features



class ArrayBuilder:
    def __init__(self, datapath):
        """
        Creates object that processes data and converts it to numpy arrays.
        Requires original PGN files to be from FICS database and in ``/data/raw``.
        Saves processed PGN in ``/data/processed`` and np arrays in ``/data/arrays``.
        
        :param datapath: path to data folder
        """
        if not os.path.exists(datapath):
            raise FileNotFoundError("create /data/raw and put data in there")

        self.paths_dict = {
            'data': datapath,
            'raw': oshelper.pathjoin(datapath, "raw"),
            'processed': oshelper.pathjoin(datapath, "processed", "processed.pgn"),
            'arrays': oshelper.pathjoin(datapath, "arrays")
        }

        oshelper.create_if_not_exists(self.paths_dict['processed'])
        oshelper.create_if_not_exists(self.paths_dict['raw'])

    def process_files(self):
        """
        Recursive function that goes through all directories in ``/data/raw``
        and processes them.
        """
        def process_level(path):
            if os.path.isfile(path):
                self._remove_metadata(path)
            else:

                for group in os.listdir(path):
                    process_level(os.path.join(path, group))

        process_level(self.paths_dict['raw'])


    def _remove_metadata(self, rawpath):

        forfeit = "forfeits by disconnection"
        with open(rawpath, 'r') as raw, \
                open(self.paths_dict['processed'], 'w+') as processed:
            for line in raw:

                if line[:2] == "1." and forfeit not in line: # Game is there
                    end = line.index('{')
                    processed_line = line[:end].replace('+', '').replace('#', '') + '\n'
                    processed_line = re.sub(r'[1-9][0-9]*\.\s', '', processed_line)

                    result = line[end:]
                    if '1-0' in result:
                        result = '0'
                    elif '0-1' in result:
                        result = '1'
                    else:
                        result = '1/2'

                    processed.write("{result} {movesequence}".format(result=result, movesequence=processed_line))


    def convert_to_arrays(self):
        """
        Converts to two arrays, X and y.
        X is the list of all chess positions in feature_list form
        Y is the list of evaluations in the form [good for white, draw, good for black]
        :return: X and y
        """
        features, labels = [], []
        color_dict = {color.white: 0, color.black: 1}
        with open(self.paths_dict['processed'], 'r') as processed:

            for i, line in enumerate(processed):

                print("On game number {}".format(i))
                if line[0:3] == "1/2":  # Game is drawn
                    result = 0.5
                    game_str = line[4:]
                else:                   # Victor exists
                    result = int(line[0:1])
                    game_str = line[2:]
                print("Result of the game was {}".format(result))

                move_list = game_str.split(' ')
                print(move_list)
                data_board = Board.init_default()
                color_itr = itertools.cycle([color.white, color.black])

                for move_str in move_list:
                    current_color = next(color_itr)
                    print(move_str)
                    print(data_board)
                    print(color_dict[current_color])

                    try:
                        move = converter.incomplete_alg(move_str, current_color)
                    except AttributeError as error:
                        print(error)
                        break
                    if move is None:
                        print("Broken in incomplete alg")
                        break

                    try:
                        move = converter.make_legal(move, data_board)
                    except AttributeError as error:
                        print(error)
                        break
                    if move is None:
                        print("Broken in make lega alg")
                        break

                    data_board.update(move)

                    features.append(extract_features(data_board))
                    if color_dict[current_color] == result: # This player won the game
                        if current_color == color.white:
                            labels.append([1, 0, 0])
                        else:
                            labels.append([0, 0, 1])

                    elif color_dict[current_color] == result: # This player lost the game
                        if current_color == color.white:
                            labels.append([0, 0, 1])
                        else:
                            labels.append([1, 0, 0])
                    else: # Game was drawn
                        labels.append([0, 1, 0])

                if i >= 3824:
                    break

        np.save(oshelper.pathjoin(self.paths_dict['arrays'], 'features'), np.array(features))
        np.save(oshelper.pathjoin(self.paths_dict['arrays'], 'labels'), np.array(labels))

        return features, labels
