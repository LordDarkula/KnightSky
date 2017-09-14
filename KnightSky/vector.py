import numpy as np

from chess_py import Board, color
from chess_py.pieces.piece_const import Piece_values


def bitmap(board):
    """
    Converts board to 1D numpy array consisting of
    piece values.
    :type: board: Board
    :rtype: np.array
    """
    fit_values = Piece_values()

    # Convert board to 1D numpy array
    return np.array([fit_values.val(square, color.white) for square in board])




