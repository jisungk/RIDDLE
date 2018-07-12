"""
test_mlp.py

Unit test(s) for the `mlp.py` module.

Requires:   pytest, NumPy, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

import pytest

import numpy as np
from riddle.models import mlp


class TestMLP():

    def test__process_data(self):
        x = [[9, 4, 5, 2], [4, 4, 5, 6, 2], [0, 2, 3, 4], [3]]
        y = [0, 2, 1, 6]
        num_feature, num_class = 10, 7

        x_expected = np.asarray(
            [[0, 0, 1, 0, 1, 1, 0, 0, 0, 1],
             [0, 0, 1, 0, 1, 1, 1, 0, 0, 0],
             [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]],
            dtype=np.float)

        y_expected = np.asarray(
            [[1, 0, 0, 0, 0, 0, 0],
             [0, 0, 1, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1]],
            dtype=np.float)

        x_out = mlp._process_x(x, num_feature=num_feature)
        y_out = mlp._process_y(y, num_class=num_class)

        assert np.all(np.equal(x_out, x_expected))
        assert np.all(np.equal(y_out, y_expected))
