import warnings

from chess_py import *

class Tree:
    def __init__(self, pos, col, depth):
        """
        Creates tree move tree to store all positions up to a certain depth.

        :param pos: Initial position for ``Tree`` to store
        :type: pos: Board

        :param col: Color of player whose ``Tree`` this is
        :type: col: Color

        :param depth: Depth of calculations
        :type: depth: int
        """
        self.position = pos
        self.color = col
        self._depth = depth

        self.head = Node(None, pos, col)
        self._build_tree()
        self.tails = [self.head]

    def _build_tree(self):
        for _ in range(self._depth):
            self.extend_tree()

    @property
    def depth(self):
        return self._depth

    @depth.setter
    def depth(self, depth):
        if depth <= 0:
            warnings.warn("Tree depth must be greater than 0")

        elif depth > self._depth:
            while depth > self._depth:
                self.extend_tree()
                self._depth += 1

        elif depth < self._depth:
            while depth < self._depth:
                self.shrink_tree()
                self.depth -= 1

    def extend_tree(self):
        """
        Adds one layer to the bottom of the tree
        """
        old_tails = self.tails
        self.tails = []

        for tail in old_tails:
            self.add_tails(tail)

    def add_tails(self, node):
        tail_moves = node.all_possible_moves(node.color)

        for move in tail_moves:
            position = node.position.cp()
            position.update(move)

            child = Node(move, position, node.color.opponent())
            node.add_child(child)

            self.tails.append(child)

    def shrink_tree(self):
        """
        Deletes leaf nodes of ``Tree`` and updates ``self.tails``
        """
        self.tails = []
        def find_tails(node):
            if node.children.children is None:
                node.children = None
                self.tails.append(node)
                return

            for child in node.children:
                find_tails(child)

        find_tails(self.head)

    def reset_tails(self):
        """
        Recursively moves through ``Tree`` and finds leaf
        nodes. Updates ``self.tails`` with found leaf nodes.
        """
        self.tails = []

        def find_tails(node):
            if node.children is None:
                self.tails.append(node)
                return

            for child in node.children:
                find_tails(child)

        find_tails(self.head)

    def generate_move(self):
        """
        Finds best move and updates tree.
        :return: best move
        """
        # TODO find best move and return it
        # TODO update tree with best move
        pass

    @staticmethod
    def best_continuation(node, val_scheme):
        board = Board.init_default()
        board.material_advantage()
        if len(node.children) == 0:
            raise Exception("No continuation")

        best_pos = node.children[0]
        advantage = best_pos.position.material_advantage(node.color, val_scheme)
        for child in node.children:
            pot_best = child.position
            pot_advantage = pot_best.position.material_advantage(node.color, val_scheme)

            if pot_advantage > advantage:
                advantage = pot_advantage
                best_pos = pot_best

        return best_pos

    def update_from_position(self, position):
        """
        Updates ``Tree`` with opponent's response to my
        previous move so I can start calculating again.

        :param position: position when it i my turn
        """
        for child in self.head.children:
            if child.position == position:
                self.head = child
                self.reset_tails()
                self.extend_tree()
                return


class Node:
    def __init__(self, move, pos, col, children=None):
        self.move = move
        self.position = pos
        self.color = col
        self.children = children or []

    def add_child(self, child):
        self.children.append(child)

    def add_children(self, children):
        self.children.extend(children)