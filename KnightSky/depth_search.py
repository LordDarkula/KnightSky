import chess_py
from chess_py.pieces.piece_const import Piece_values
from chess_py import color
from chess_py.core.algebraic import notation_const
from chess_py import Player


class Ai(Player):
    def __init__(self, input_color):
        """
        Creates interface for human player.

        :type input_color: Color
        """
        self.piece_scheme = Piece_values()
        self.my_moves = []
        super(Ai, self).__init__(input_color)

    def generate_move(self, position):
        """
        Returns valid and legal move given position
        
        :type position: Board
        :rtype Move
        """
        print(position)
        print("Running depth search")
        
        self.my_moves = position.all_possible_moves(self.color)

        move = self.depthSearch(position, 3, self.color)
        print("Final advantage ", move[1])
        return move[0]

    def is_quiet_position(self, input_color, position):
        print("is quiet running")
        
        enemy_moves = position.all_possible_moves(input_color.opponent())

        for move in self.my_moves:
            if move.status == notation_const.CAPTURE or \
                    move.status == notation_const.CAPTURE_AND_PROMOTE:
                return False

        for move in enemy_moves:
            if move.status == notation_const.CAPTURE or \
                    move.status == notation_const.CAPTURE_AND_PROMOTE:
                return False

        return True

    def best_move(self, position, color):
        """
        Finds the best move based on material after the move
        
        :type position: Board
        :type color: Color
        :rtype: Move
        """

        moves = position.all_possible_moves(input_color=color)

        my_move = moves[0]
        advantage = position.advantage_as_result(my_move, self.piece_scheme)

        for move in moves:
            test = position.copy()
            test.update(move)
            # print("In the best move for")
            pot_advantage = test.material_advantage(move.color, self.piece_scheme)

            if len(test.all_possible_moves(color.opponent())) == 0:
                return move, 100

            if pot_advantage > advantage:

                my_move = move
                advantage = pot_advantage

        return my_move, advantage

    def best_reply(self, move, position):
        """
        Finds the best move based on material after the move
        
        :type move: Move
        :type position: Board
        :rtype: Move
        """
        test = position.copy()
        test.update(move)
        reply = self.best_move(test, move.color.opponent())
        return reply, test.advantage_as_result(reply[0], self.piece_scheme)



#TODO if pot_worst is the same as worst decide which move is better for me
#TODO safegard against checkmate
#TODO build move tree to avoid long wait

    def depthSearch(self, position, depth, color):
        """
        Returns valid and legal move given position
        
        :type position: Board
        :type depth: int
        :type color: Color
        :rtype: Move
        """
        print("Depth: ", depth)
        if depth == 1 or self.is_quiet_position(color, position):
            return self.best_move(position, color)

        moves = position.all_possible_moves(color)
        print("Number of possible moves ", len(moves))

        my_move = None
        for move in moves:

            print(move)
            test = position.copy()
            test.update(move)

            # Checks for checkmate
            if len(test.all_possible_moves(color.opponent())) == 0:
                return move, 100


            best_reply = self.depthSearch(test, depth=depth - 1, color=color.opponent())

            print(best_reply[0])
            print("My Advantage", -best_reply[1])
            if my_move is None or my_move[1] < -best_reply[1]:
                my_move = move, -best_reply[1]

            print("New line")
        return my_move

    def weed_checkmate(self, moves, position):
        print("weeding 0ut checkmate")
        final = []
        for i in range(len(moves)):
            test = position.copy()
            test.update(moves[i])

            if not self.best_reply(moves[i], position)[1] == 100:
                final.append(moves[i])

        if len(final) == 0:
            final.append(moves[0])

        return final


