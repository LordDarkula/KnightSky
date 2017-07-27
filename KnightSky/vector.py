import numpy as np
import math

from chess_py import Board, color
from chess_py.pieces.piece_const import Piece_values

def boardtovector(board):
    fit_values = Piece_values()

    # Convert board to 1D numpy array
    bitboard = np.array([fit_values.val(square, color.white) for square in board])


    # TODO reshape array



