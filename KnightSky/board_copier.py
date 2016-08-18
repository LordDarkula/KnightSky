from chess_py import Board
from copy import copy as cp

def copy(board):
    """
    Copies the board faster than deepcopy()
    :type board Board
    :rtype Board
    """
    return Board([[cp(piece) for piece in board.position[index]] for index, row in enumerate(board.position)])
