# -*- coding: utf-8 -*-

import os
import numpy as np
import re
import itertools
import random

from chess_py import *
from KnightSky.vector import bitmap


original_data = "ficsgamesdb.pgn"
original_path = original_data # os.path.join(os.pardir, os.pardir, original_data)
print(original_path)

processed_data = "processed.pgn"
processed_path = processed_data # os.path.join(os.pardir, os.pardir, processed_data)

print(processed_path)


def remove_metadata():

    forfeit = "forfeits by disconnection"
    with open(original_path, 'r') as raw, open(processed_path, 'w+') as processed:
        for line in raw:
            if line[:2] == "1." and forfeit not in line:
                end = line.index('{')
                processed_line = line[:end].replace('+', '').replace('#', '') + '\n'
                processed_line = re.sub(r'[1-9][0-9]*\.\s', '', processed_line)

                if '1O-O-O' in processed_line:
                    print("This is the line " + line)
                    print("Processed " + processed_line)
                processed.write(processed_line)


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
