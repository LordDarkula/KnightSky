from chess_py import *

class Tree:
    def __init__(self, pos, col, depth):
        """

        :param pos: Initial position for ``Tree`` to store
        :type: pos: Board

        :param col: Color of player whose ``Tree`` this is
        :type: col: Color

        :param depth: Depth of calculations
        :type: depth: int
        """
        self.position = pos
        self.color = col
        self.depth = depth

        self.head = Node(None, pos, col)
        self._build_tree()
        self.tails = [self.head]

    def _build_tree(self):
        for _ in range(self.depth):
            self.extend_tree()

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

    def update(self, response):
        """
        Updates ``Tree`` with opponent's response to my
        previous move so I can start calculating again.

        :param response:
        """
        for child in self.head.children:
            if child.move == response:
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