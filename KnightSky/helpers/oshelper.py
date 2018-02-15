# -*- coding: utf-8 -*-

"""
Helpers to make widely used os tasks simpler.
"""
import os


def create_if_not_exists(path, is_file=False):
    """
    Creates directory or file if it does not exist.

    :param path: Path to file
    :type: path: str
    """
    path = os.path.normpath(abspath(path))
    if not os.path.exists(path):
        print("This is it " + path)
        if is_file:  # Path refers to a file
            print("path is not dir")
            os.makedirs(os.path.dirname(path))
            open(path, 'w').close()
        else:
            print("path is dir")
            os.makedirs(path)


""" Aliases """


def abspath(path):
    """
    Copy of ``os.path.abspath`` to make the function name shorter.

    :param: path: Path to file
    :type: path: str
    """
    return os.path.abspath(path)


def pathjoin(*path):
    """
    Copy of ``os.path.join`` to make the function name shorter.

    :param: path: Path to file
    :type: path: str
    """
    return os.path.join(*path)
