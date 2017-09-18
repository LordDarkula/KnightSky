# -*- coding: utf-8 -*-
"""
Script meant to automate the naming of individual tensorboard runs.
Calling ``current_run_directory()`` will create the necessary directories if they do not exist,
increment the run count, and return the path to the current tensorboard run.
"""

import os
from KnightSky.helpers import oshelper


def current_run_directory(run_name='knight'):
    """
    Sets correct tensorboard directory dynamically based on number of runs.
    Delete tensorboard directory to start count over
    """
    tbdir = _construct_tb_path()
    count = _incr_tb_run_number(tbdir, run_name)
    oshelper.create_if_not_exists(oshelper.pathjoin(tbdir, run_name))
    return oshelper.pathjoin(tbdir, run_name, count)


def _construct_tb_path():
    """ Creates correct absolute path from relative path of tensorboard directory. """
    tbpathlist = [os.pardir for _ in range(3)]
    tbpath = oshelper.pathjoin(*tbpathlist)
    tbpath = oshelper.pathjoin(tbpath, "tensorboard")
    tbpath = oshelper.abspath(tbpath)
    oshelper.create_if_not_exists(tbpath)
    return tbpath


def _incr_tb_run_number(tbpath, run_name):
    """
    Increments tensorboard count by 1 for the new run. If no runs are present, create the directory itself.
    :param tbpath: path of tensorboard
    :return: The number of runs plus 1 for use as the name of the new run
    """
    filename = run_name + '.txt'
    run_number_path = oshelper.pathjoin(tbpath, filename)
    if not os.path.exists(run_number_path):

        with open(run_number_path, 'w') as f:
            f.write('1')

        return '1'

    else:
        with open(run_number_path, 'r') as f:
            current = int(f.readline())

        with open(run_number_path, 'w') as f:
            f.write(str(current + 1))

        return str(current + 1)
