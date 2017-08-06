import numpy as np


def material_y(bitmap_X, bitmap_y):
    bitmap_X = list(bitmap_X)
    final_X, final_y = [], []

    for i, position in enumerate(bitmap_X):

        material = 0
        for square in position:
            material += square

        if material == 0:
            print("Even")
            continue

        final_X.append(position)
        if material > 1:
            print("white {}".format(material))
            final_y.append([1, 0])

        elif material < -1:
            print("black {}".format(material))
            final_y.append([0, 1])

        else:
            continue

    return np.array(final_X), np.array(final_y)
