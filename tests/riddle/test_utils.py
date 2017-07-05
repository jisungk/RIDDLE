"""
test_utils.py

Unit test(s) for the `utils.py` module.

Requires:   pytest, NumPy, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

import pytest

import sys; sys.dont_write_bytecode = True
import os

from numpy import random

from riddle.utils import split_data

class TestUtils():
    def test_split_data(self):
        X, y = [], []

        N = random.randint(10, 1000)

        for i in range(N):
            X.append(random.rand(random.randint(4, 100)).tolist())
            y.append(random.randint(0, 10))

        perm_indices = random.permutation(N)

        k = 10
        for k_idx in range(k):
            (X_train, y_train), (X_test, y_test) = split_data(X, y, 
                k_idx=k_idx, k=10, perm_indices=perm_indices)

            assert sorted(X) == sorted(X_train + X_test)
            assert sorted(y) == sorted(y_train + y_test)

        (X_train_1, y_train_1), (X_test_1, y_test_1) = split_data(X, y, 
            k_idx=k_idx, k=10, perm_indices=perm_indices)
        
        (X_train_2, y_train_2), (X_test_2, y_test_2) = split_data(X, y, 
            k_idx=k_idx, k=10, perm_indices=perm_indices)

        assert len(X_train_1) == len(X_train_2)
        for idx in range(len(X_train_1)):
            assert X_train_1[idx] == X_train_2[idx]
            assert y_train_1[idx] == y_train_2[idx]

        assert len(X_test_1) == len(X_test_2)
        for idx in range(len(X_test_1)):
            assert X_test_1[idx] == X_test_2[idx]
            assert y_test_1[idx] == y_test_2[idx]
