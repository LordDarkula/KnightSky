# -*- coding: utf-8 -*-

import os
import numpy as np
import re
import itertools

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

            if i > 50:
                break

    bitmap_X = np.array(bitmap_X)
    bitmap_y = np.array(bitmap_y)

    np.save('bitmap_X', bitmap_X)
    np.save('bitmap_y', bitmap_y)

    return bitmap_X, bitmap_y


if __name__ == '__main__':
    remove_metadata()
    x, y = convert_to_arrays()
    print(x)
    print(y)