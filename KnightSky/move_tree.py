import warnings
from copy import copy as cp

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
        self.tails = [self.head]
        self._build_tree()

    def _build_tree(self):
        for _ in range(self._depth):
            self.extend_tree()

        self._depth = int(self._depth / 2)

    @property
    def depth(self):
        return self._depth

    def extend_tree(self):
        """
        Adds one layer to the bottom of the tree
        """
        print("Extending tree")
        old_tails = self.tails
        self.tails = []

        for tail in old_tails:
            self._add_tails(tail)

        self._depth += 1

    def _add_tails(self, node):

        for move in node.position.all_possible_moves(node.color):
            position = cp(node.position)
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
            if node.children[0].children is []:
                node.children = []
                self.tails.append(node)
                return

            for child in node.children:
                find_tails(child)

        find_tails(self.head)

        self._depth -= 1

    def reset_tails(self):
        """
        Recursively moves through ``Tree`` and finds leaf
        nodes. Updates ``self.tails`` with found leaf nodes.
        """
        self.tails = []

        def find_tails(node):
            if node.is_tail:
                self.tails.append(node)
                return

            for child in node.children:
                find_tails(child)

        find_tails(self.head)

    @staticmethod
    def best_continuation(node, val_scheme):
        if node.is_tail:
            raise Exception("No continuation")

        if len(node.children) == 1:
            return node.children[0]

        return max(*node.children, key=lambda x: x.position.material_advantage(node.color, val_scheme))

    def update_from_position(self, position):
        """
        Updates ``Tree`` with opponent's response to my
        previous move so I can start calculating again.

        :param position: position when it is my turn
        """
        for child in self.head.children:
            if child.position == position:
                self.update_from_node(child)
                print("worked")
                return

        print("did not work")

    def update_from_node(self, node):
        self.head = node
        self._depth -= 1
        self.reset_tails()
        self.extend_tree()


class Node:
    def __init__(self, move, pos, col, children=None):
        self.move = move
        self.position = pos
        self.color = col
        self._children = children or []

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.move)

    @property
    def children(self):
        return self._children

    @property
    def is_tail(self):
        return len(self._children) == 0

    def add_child(self, child):
        self._children.append(child)

    def add_children(self, children):
        self._children.extend(children)