"""
Processes data stored in ``../../data/raw`` and converts them
to numpy arrays to be stored in ``../../data/arrays`` as
``features.npy`` and ``labels.npy``
"""

import os

from KnightSky.preprocessing import arraybuilder


if __name__ == '__main__':
    proc = arraybuilder.ArrayBuilder(os.path.abspath(os.path.join(os.pardir, os.pardir, 'data')))
    proc.process_files()
    proc.convert_to_arrays()
