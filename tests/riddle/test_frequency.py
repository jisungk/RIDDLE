"""
test_frequency.py

Unit test(s) for the `frequency.py` module.

Requires:   pytest, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

import pytest

import sys; sys.dont_write_bytecode = True

from riddle.frequency import *

def test_get_frequency_table():
    X = [[0] * 20, [1] * 5, [1, 2, 1], \
        [2, 1, 1], [1] * 3]
    y = [0, 1, 1, 0, 0]
    idx_feat_dict = {0: 'Matthew', 1:'James', 2:'Andrew'}
    idx_class_dict = {0:'A', 1:'B'}
    
    freq_table = get_frequency_table(X, y, idx_feat_dict, idx_class_dict)
    expected_freq_table = [[20, 5, 1], [0, 7, 1]]
    assert np.all(freq_table  == expected_freq_table)