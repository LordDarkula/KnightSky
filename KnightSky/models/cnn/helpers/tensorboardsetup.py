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
    Returns correct tensorboard directory which is set dynamically based on number of runs.
    Delete tensorboard directory to start count over
    """
    runpath = oshelper.pathjoin(tmp_path, run_name)
    print(runpath)
    oshelper.create_if_not_exists(runpath, is_file=False)
    count = _increment_tb_run_number(runpath)
    return oshelper.pathjoin(runpath, str(count))


def _increment_tb_run_number(runpath):
    """
    Increments tensorboard count by 1 for the new run. If no runs are present, create the directory itself.
    :param runpath: path of tensorboard
    :return: The number of runs plus 1 for use as the name of the new run
    """
    countfilename = oshelper.pathjoin(runpath, 'count.txt')
    if os.path.exists(countfilename):
        # Open count.txt where number of runs are stored
        with open(countfilename, 'r') as f:
            current = int(f.readline())

        with open(countfilename, 'w') as f:
            f.write(str(current + 1))

        return current + 1

    else:
        # Create new file file to store tensorboard runs if none exists
        with open(countfilename, 'w') as f:
            f.write('1')

        return 1
