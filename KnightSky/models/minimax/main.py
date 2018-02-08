from chess_py import *
from KnightSky.models.minimax.depth_search import Ai

game = Game(Human(color.white), Ai(color.black))
game.play()

