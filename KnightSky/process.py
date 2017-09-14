# -*- coding: utf-8 -*-

import os
import numpy as np
import re
import itertools
import random

from chess_py import *
from KnightSky.vector import bitmap


original_path = os.path.abspath("../data/raw")
print("raw data in {}".format(original_path))

processed_path = os.path.abspath("../data/processed")
processed_filename = "processed.pgn"
if not os.path.exists(processed_path):
    os.makedirs(processed_path)
print("Processed data to be placed in {}".format(processed_path))


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
            open(os.path.join(processed_path, processed_filename), 'w+') as processed:
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
    bitmap_X, bitmap_y = [], []
    color_dict = {color.white: 0, color.black: 1}
    with open(processed_path, 'r') as processed:
        for i, line in enumerate(processed):
            print("On game number {}".format(i))
            move_list = line.split(' ')
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
                    break

                try:
                    move = converter.make_legal(move, data_board)

                except AttributeError as error:
                    print(error)
                    break

                if move is None:
                    break

                data_board.update(move)

                bitmap_X.append(bitmap(data_board))
                bitmap_y.append(color_dict[current_color])

            if i >= 3824:
                break

    bitmap_X = np.array(bitmap_X)
    bitmap_y = one_hot(bitmap_y)

    np.save('bitmap_X', bitmap_X)
    np.save('bitmap_y', bitmap_y)

    return bitmap_X, bitmap_y


def one_hot(vector):
    def hot_or_not(i, j):
        return 1 if i == j else 0
    return np.array([[int(hot_or_not(i, j)) for j in range(2)] for i in list(vector)])


def randomly_assign_train_test(X_data, y_data, test_size=0.1):
    data = list(zip(list(X_data), list(y_data)))
    random.shuffle(data)

    shuffled_X, shuffled_y = zip(*data)

    X_split_index = int(len(shuffled_X) * test_size)
    y_split_index = int(len(shuffled_y) * test_size)

    X_train, X_test = shuffled_X[X_split_index:], shuffled_X[:X_split_index]
    y_train, y_test = shuffled_y[y_split_index:], shuffled_y[:y_split_index:]
    return X_train, X_test, y_train, y_test


def next_batch(X, y, batch_size=100):
    n_batches = int(len(X) / batch_size)

    for batch in range(n_batches):
        start = (batch * batch_size)
        end = start + batch_size if start + batch_size < len(X) else len(X) - 1
        yield X[start:end], y[start:end]


if __name__ == '__main__':
    process_files(original_path)