# -*- coding: utf-8 -*-

"""
Chess playing program
Everything starts here


8 ║♜♞♝♛♚♝♞♜
7 ║♟♟♟♟♟♟♟♟
6 ║…………………………………
5 ║…………………………………
4 ║…………………………………
3 ║…………………………………
2 ║♙♙♙♙♙♙♙♙
1 ║♖♘♗♕♔♗♘♖
--╚═══════════════
——-a b c d e f g h

Copyright © 2016 Aubhro Sengupta. All rights reserved.
"""
import numpy as np
from chess_py import *

from KnightSky.depth_search import Ai
from KnightSky.process import remove_metadata, convert_to_arrays
from KnightSky.train_model import create_model, run_model


def main():
    """
    Main method
    """
    print("Creating a new game...")

    new_game = Game(Human(color.white), Ai(color.black))
    result = new_game.play()

    print("Result is ", result)

if __name__ == "__main__":
    # remove_metadata()
    # x, y = convert_to_arrays()

    bitmap_X = np.load('bitmap_X.npy')
    bitmap_y = np.load('bitmap_y.npy')

    print(bitmap_X[:10])
    print(bitmap_y[:10])

    optimizer, accuracy = create_model()
    run_model(optimizer, accuracy, bitmap_X, bitmap_y)
