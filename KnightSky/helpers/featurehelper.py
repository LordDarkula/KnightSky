# -*- coding: utf-8 -*-

"""
Helpers to make constructing features and labels easier.
"""
import numpy as np

from chess_py import color
from chess_py.pieces.piece_const import Piece_values


def features(board):
    """
    Converts board to 1D numpy array consisting of
    piece values.
    :type: board: Board
    :rtype: np.array
    """
    fit_values = Piece_values()

    # Convert board to 1D numpy array
    return np.array([fit_values.val(square, color.white) for square in board])


def material_y(bitmap_X):
    """
    Creates one hot vectors 
    [0, 0, 1] if white has more material,
    [1, 0, 0] if black has more material,
    [0, 1, 0] if the material is even.
    
    :param bitmap_X: list of all positions to create labels for
    :return: features, labels
    """
    bitmap_X = list(bitmap_X)
    final_X, final_y = [], []

    for i, position in enumerate(bitmap_X):

        material = 0
        for square in position:
            material += square

        if material == 0:
            print("Even")
            continue

        final_X.append(position)
        if material > 1:
            print("white {}".format(material))
            final_y.append([1, 0, 0])

        elif material < -1:
            print("black {}".format(material))
            final_y.append([0, 0, 1])

        else:
            print("Material even")
            final_y.append([0, 1, 0])

    return np.array(final_X), np.array(final_y)
