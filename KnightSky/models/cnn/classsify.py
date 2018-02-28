# -*- coding: utf-8 -*-

"""
Classification functions create one hot vectors
[0, 0, 1] if white has more material,
[1, 0, 0] if black has more material,
[0, 1, 0] if the material is even.
"""

import numpy as np


def material(positions):
    """
    Creates one hot vectors by materials

    :param positions: list of all positions to create labels for
    :return: features, labels
    """
    advantages = np.zeros((3, len(positions)), dtype=int)

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


def turn(games):
    for game in games['games']:
        pass


def result(games):
    for game in games['games']:
        pass
