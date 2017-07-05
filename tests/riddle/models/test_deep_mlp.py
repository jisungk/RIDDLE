"""
test_deep_mlp.py

Unit test(s) for the `deep_mlp.py` module.

Requires:   pytest, NumPy, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

import pytest
import sys; sys.dont_write_bytecode = True

import numpy as np
from riddle.models.deep_mlp import process_X_data, process_y_data

class TestDeepMLP():
    def test_process_data(self):
        X = [[9, 4, 5, 2], [4, 4, 5, 6, 2], [0, 2, 3, 4], [3]]
        y = [0, 2, 1, 6]
        nb_features, nb_classes = 10, 7

        X_expected = np.asarray([[0, 0, 1, 0, 1, 1, 0, 0, 0, 1], 
            [0, 0, 1, 0, 1, 1, 1, 0, 0, 0], [1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]], dtype=np.float)

        y_expected = np.asarray([[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0], 
            [0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1]], dtype=np.float)

        X_out = process_X_data(X, nb_features=nb_features)
        y_out = process_y_data(y, nb_classes=nb_classes)

        assert np.all(np.equal(X_out, X_expected))
        assert np.all(np.equal(y_out, y_expected))
    