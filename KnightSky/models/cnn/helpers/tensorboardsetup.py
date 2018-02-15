# -*- coding: utf-8 -*-
"""
Script meant to automate the naming of individual tensorboard runs.
Calling ``current_run_directory()`` will create the necessary directories if they do not exist,
increment the run count, and return the path to the current tensorboard run.
"""

import os
from KnightSky.helpers import oshelper


class TensorboardManager:
    def __init__(self, tmp_path, run_name='knight', run_tracking_file='count.txt'):
        """
        Returns correct tensorboard directory which is set dynamically based on number of runs.
        Delete tensorboard directory to start count over
        """
        self._path = oshelper.pathjoin(tmp_path, run_name)
        self._run_tracking_file_path = oshelper.pathjoin(self._path, run_tracking_file)
        print(self._path)

        if not os.path.exists(self._run_tracking_file_path):
            with open(self._run_tracking_file_path, 'w') as f:
                f.write(str(1))

    @property
    def tensorboard_path(self):
        return oshelper.pathjoin(self._path, self.run_number)

    @property
    def run_number(self):
        with open(self._run_tracking_file_path, 'r') as f:
            return int(f.readline())

    def __iadd__(self, other):
        """
        Increments tensorboard count by 1 for the new run. If no runs are present, create the directory itself.
        :param other: integer describing run increment size
        """
        if not isinstance(other, int):
            raise TypeError("Must increment run by integer amount")

        with open(self._run_tracking_file_path, 'r') as f:
            current_run = int(f.readline())

        with open(self._run_tracking_file_path, 'w') as f:
            f.write(str(current_run + 1))
