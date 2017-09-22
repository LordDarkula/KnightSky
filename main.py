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

import os

from KnightSky.preprocessing.arraybuilder import ArrayBuilder


if __name__ == "__main__":
    proc = ArrayBuilder(os.path.join(os.getcwd(), 'data'))
    proc.process_files()
    proc.convert_to_arrays()


