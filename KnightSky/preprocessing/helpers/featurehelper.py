# -*- coding: utf-8 -*-

"""
Helpers to make constructing features and labels easier.
"""

import numpy as np

from chess_py import color
from chess_py.pieces.piece_const import Piece_values


def extract_features_from_positions(board):
    """
    Converts board to 1D numpy array consisting of
    piece values.
    :type: board: Board
    :rtype: np.array
    """
    fit_values = Piece_values()

    # Convert board to 1D numpy array
    return np.array([fit_values.val(square, color.white) for square in board])


def classify_position_by_material(positions):
    """
    Creates one hot vectors by materials
    [0, 0, 1] if white has more material,
    [1, 0, 0] if black has more material,
    [0, 1, 0] if the material is even.
    
    :param positions: list of all positions to create labels for
    :return: features, labels
    """
    advantages = np.zeros((len(positions)), dtype=int)

    for i, position in enumerate(positions):

        material_imbalance = np.sum(position)

        if material_imbalance > 1:
            print("white {}".format(material_imbalance))
            advantages[i][0] = 1

        elif material_imbalance < -1:
            print("black {}".format(material_imbalance))
            advantages[i][2] = 1

        else:
            print("Material even")
            advantages[i][1] = 1

    return advantages
