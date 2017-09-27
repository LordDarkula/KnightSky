# -*- coding: utf-8 -*-
"""
Script meant to automate the naming of individual tensorboard runs.
Calling ``current_run_directory()`` will create the necessary directories if they do not exist,
increment the run count, and return the path to the current tensorboard run.
"""

import os
from KnightSky.helpers import oshelper


def current_run_directory(tmp_path, run_name='knight'):
    """
    Sets correct tensorboard directory dynamically based on number of runs.
    Delete tensorboard directory to start count over
    """
    runpath = oshelper.pathjoin(tmp_path, run_name)
    print(runpath)
    oshelper.create_if_not_exists(runpath)
    count = _incr_tb_run_number(runpath)
    return oshelper.pathjoin(runpath, count)


def _incr_tb_run_number(runpath):
    """
    Increments tensorboard count by 1 for the new run. If no runs are present, create the directory itself.
    :param tbpath: path of tensorboard
    :return: The number of runs plus 1 for use as the name of the new run
    """
    countfilename = oshelper.pathjoin(runpath, 'count.txt')
    if not os.path.exists(countfilename):

        with open(countfilename, 'w') as f:
            f.write('1')

        return '1'

    else:
        with open(countfilename, 'r') as f:
            current = int(f.readline())

        with open(countfilename, 'w') as f:
            f.write(str(current + 1))

        return str(current + 1)
