#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common tools"""

import os
import sys

sys.dont_write_bytecode = True

from time import time
from functools import wraps

def mkdir(directory):
    """create new folder if ``directory`` doesn't exists

    Args:
        directory (:obj:`str`): the string of the new folder

    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def make_case_dir(folder_name, noise_level):
    """create new folder with noise level appended

    Args:
        folder_name (:obj:`str`): the string of the new folder.
        noise_level (:obj:`int`): SNR ratio.
    """
    mkdir('./result')
    mkdir('./result/' + folder_name + '_noise_level_' + str(noise_level))
  
    return


def timing(func_to_be_timed):
    """
    timing decorator

    :type func_to_be_timed: function
    :param func_to_be_timed: function that is to be timed
    """
    @wraps(func_to_be_timed)
    def wrapper(*args, **kwargs):
        start = time()
        result = func_to_be_timed(*args, **kwargs)
        end = time()
        print('Elapsed time: {}'.format((end-start)/60.0) + ' mins ')
        return result

    return wrapper


