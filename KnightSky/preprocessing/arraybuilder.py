# -*- coding: utf-8 -*-

"""
Class to access all pre-processing functionality.
"""

import os
import numpy as np
import re
import json
from chess_py import *

from definitions import ROOT_DIR
from KnightSky.helpers import oshelper
from KnightSky.preprocessing.helpers import featurehelper


class ArrayBuilder:
    """
    ``ArrayBuilder`` processes data and converts it into json, then numpy arrays.
    Requires original PGN files to be from FICS database and in ``data/raw``.
    Saves json in ``data/processed`` and numpy arrays in ``data/arrays``.
    """
    def __init__(self, datapath):
        """
        Builds dictionary to store paths and create output directories in ``datapath`` if
        they do not exist.

        :param: datapath: path to data folder
        :type: datapath: str

        :raise: FileNotFoundError: if datapath is invalid
        """
        if not os.path.exists(datapath):
            raise FileNotFoundError("{} not found. "
                                    "Please create /raw path and put chess data in there".format(datapath))

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

    @staticmethod
    def _moves_to_positions_and_labels(move_sequence, result):
        """
        Converts sequence of moves stored in algebraic notation to array of positions and advantages
        by playing those moves and recording the state of the board after each move.
        Board initialized in a default state.

        :param move_sequence: sequence of moves in algebraic notation
        :type move_sequence: Iterable[str]

        :param result: result of game (0, 1, or 2)
        :type result: int

        :return: array of 1D board features, 1D array of results of either 0, 1, 2
        :rtype: np.array
        """
        processing_board = Board.init_default()
        current_color = color.white
        piece_id = piece_const.PieceValues.init_manual(PAWN_VALUE=1,
                                                       KNIGHT_VALUE=2,
                                                       BISHOP_VALUE=3,
                                                       ROOK_VALUE=4,
                                                       QUEEN_VALUE=5)
        positions = np.zeros([len(list(move_sequence)), 64], dtype=np.int)
        advantages = np.zeros([len(move_sequence)], dtype=np.int)
        symbols = ''

        for index, algebraic_move_str in enumerate(move_sequence):
            move = converter.incomplete_alg(algebraic_move_str, current_color, processing_board)
            processing_board.update(move)
            symbols += move.piece.symbol

            positions[index] = featurehelper.extract_features_from_position(processing_board, piece_id)
            advantages[index] = result

            current_color = current_color.opponent()

            print(symbols)

        print(processing_board)
        return positions, advantages

    def convert_to_arrays(self, split_games=False):
        """
        Converts json dict of moves in algebraic notation for a set of games to features and labels
        to feed into a neural network.

        :param split_games: Specify whether to split up features and labels by game.
        Useful for recurrent neural networks.
        :type split_games: bool

        ``features`` is the list of all chess positions in the form
        [number of games, 64 (number of squares on the board)]

        ``labels`` is the list of evaluations in the form
        [number of games, 3 (good for white, draw, good for black)]

        ``labels`` examples:
        [0, 0, 1] if white is doing better
        [1, 0, 0] if black is doing better
        [0, 1, 0] if the game is even.

        Saves output in ``data/arrays`` and returns it.

        :return: features, labels
        :rtype: tuple[np.array, np.array] or if split_games == True, tuple[list[np.array], list[np.array]]
        """
        with open(self.paths_dict['processed'], 'r') as f:
            self.games = json.load(f)
            number_of_games = len(self.games['games'])

            features, labels = [], []
            for index, game_dict in enumerate(self.games['games']):
                print("Game {} of {}".format(index, number_of_games))
                features_and_labels = self._moves_to_positions_and_labels(game_dict['moves'], game_dict['result'])
                features.append(features_and_labels[0])
                labels.append(features_and_labels[1])
                print()

            if not split_games:
                features = np.concatenate(features, axis=0)
                labels = np.concatenate(labels, axis=0)

        np.save(oshelper.pathjoin(self.paths_dict['arrays'], 'features-{}'
                                  .format('split' if split_games else 'combined')), features)
        np.save(oshelper.pathjoin(self.paths_dict['arrays'], 'labels-{}'
                                  .format('split' if split_games else 'combined')), labels)

        return features, labels


if __name__ == '__main__':
    builder = ArrayBuilder(os.path.join(ROOT_DIR, 'data'))
    builder.process_files()
    arrays_combined = builder.convert_to_arrays(split_games=False)
    arrays_split = builder.convert_to_arrays(split_games=True)
    print(arrays_combined[0].shape)
    print(arrays_combined[1].shape)
    print(len(arrays_split[0]))
    print(len(arrays_split[1]))
