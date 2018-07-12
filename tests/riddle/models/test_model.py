"""
test_model_utils.py

Unit test(s) for the `model_utils.py` module.

Requires:   pytest, NumPy, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

import pytest

import os
from math import fabs

import numpy as np

from riddle.models.model import chunks


class TestModel():

    def test_chunks(self):
        list_A = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        list_B = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

        for i, (a, b) in enumerate(zip(chunks(list_A, 4), chunks(list_B, 4))):
            if i == 0:
                assert a == [1, 2, 3, 4]
                assert b == [10, 9, 8, 7]
            if i == 1:
                assert a == [5, 6, 7, 8]
                assert b == [6, 5, 4, 3]
            if i == 2:
                assert a == [9, 10]
                assert b == [2, 1]

    # TODO(jisungkim): write more tests
