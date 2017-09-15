# -*- coding: utf-8 -*-

"""
Helpers to make widely used os tasks simpler.
"""
import os


def create_if_not_exists(path):
    """ Creates directory or file if it does not exist. """
    if not os.path.exists(path):
        path = abspath(path)
        if os.path.isdir(path):
            os.makedirs(path)
        elif os.path.isfile(path):
            os.makedirs(os.path.dirname(path))
            open(path, 'r').close()


""" Aliases """
def abspath(path):
    """ Copy of ``os.path.abspath`` to make the function name shorter. """
    return os.path.abspath(path)

def pathjoin(*path):
    """ Copy of ``os.path.join`` to make the function name shorter. """
    return os.path.join(*path)