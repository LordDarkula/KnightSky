import os
from KnightSky.helpers import oshelper


def current_run_directory():
    """
    Sets correct tensorboard directory dynamically based on number of runs.
    Delete tensorboard directory to start count over
    """
    tbdir = _construct_tb_path()
    count = _incr_tb_run_number(tbdir)
    return oshelper.pathjoin(tbdir, "knight", count)


def _construct_tb_path():
    """ Creates correct absolute path from relative path of tensorboard directory. """
    tbpathlist = [os.pardir for _ in range(3)]
    tbpath = oshelper.pathjoin(*tbpathlist)
    tbpath = oshelper.pathjoin(tbpath, "tensorboard")
    tbpath = oshelper.abspath(tbpath)
    return tbpath


def _incr_tb_run_number(tbpath):
    """
    Increments tensorboard count by 1 for the new run. If no runs are present, create the directory itself.
    :param tbpath: path of tensorboard
    :return: The number of runs plus 1 for use as the name of the new run
    """
    filename = "count.txt"
    run_number_path = oshelper.pathjoin(tbpath, filename)
    if not os.path.exists(run_number_path):
        os.makedirs(tbpath)
        f = open(run_number_path, 'w')
        f.write('1')
        f.close()
        return '1'

    else:
        with open(run_number_path, 'r') as f:
            current = int(f.readline())

        with open(run_number_path, 'w') as f:
            f.write(str(current + 1))

        return str(current + 1)



