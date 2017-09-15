# -*- coding: utf-8 -*-

import os
import numpy as np
import re
import itertools
from chess_py import *

from KnightSky.helpers import oshelper
from KnightSky.helpers.featurehelper import bitmap


""" Defines paths to raw data, processed data, and numpy arrays.
Creates directories if they do not exist. """
oshelper.create_if_not_exists("../data")

original_path = oshelper.abspath("../data/raw")
print("raw data in {}".format(original_path))

processed_path = os.path.abspath("../data/processed/processed.pgn")
oshelper.create_if_not_exists(processed_path)
print("Processed data to be placed in {}".format(processed_path))

arraypath = os.path.abspath("../data/arrays")
oshelper.create_if_not_exists(arraypath)
print("Arrays to be placed in {}".format(arraypath))


def process_files(path):
    if os.path.isfile(path):
        print(path)
        _remove_metadata(path)
    else:
        for group in os.listdir(path):
            process_files(os.path.join(path, group))


def _remove_metadata(path):

    forfeit = "forfeits by disconnection"
    with open(path, 'r') as raw, \
            open(processed_path, 'w+') as processed:
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

                processed.write("{} {}".format(result, processed_line))


def convert_to_arrays():
    """
    Converts to two arrays, X and y.
    X is the list of all chess positions in bitmap form
    Y is the list of evaluations in the form [good for white, draw, good for black]
    :return: X and y
    """
    bitmap_X, bitmap_y = [], []
    color_dict = {color.white: 0, color.black: 1}
    with open(processed_path, 'r') as processed:

        for i, line in enumerate(processed):

            print("On game number {}".format(i))
            if line[0:3] == "1/2": # Game is drawn
                result = 0.5
                game_str = line[4:]
            else:
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

                bitmap_X.append(bitmap(data_board))
                if color_dict[current_color] == result: # This player won the game
                    if current_color == color.white:
                        bitmap_y.append([1, 0, 0])
                    else:
                        bitmap_y.append([0, 0, 1])

                elif color_dict[current_color] == result: # This player lost the game
                    if current_color == color.white:
                        bitmap_y.append([0, 0, 1])
                    else:
                        bitmap_y.append([1, 0, 0])
                else: # Game was drawn
                    bitmap_y.append([0, 1, 0])

            if i >= 3824:
                break

    bitmap_X = np.array(bitmap_X)
    bitmap_y = np.array(bitmap_y)

    np.save(oshelper.pathjoin(arraypath, 'bitmap_X'), bitmap_X)
    np.save(oshelper.pathjoin(arraypath, 'bitmap_y'), bitmap_y)

    return bitmap_X, bitmap_y


if __name__ == '__main__':
    process_files(original_path)
    convert_to_arrays()