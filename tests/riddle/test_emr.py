"""
test_emr.py

Unit test(s) for the `emr.py` module.

Requires:   pytest, NumPy, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

import pytest

import sys; sys.dont_write_bytecode = True
import os

import numpy as np

from riddle.emr import *

class TestEMR():
    def test_read_file(self):
        temp_fn = 'ut_emr_data.temp'

        with open(temp_fn, 'w+') as f:
            f.write('test line 1\n test line 2\ntest line 3 \nthank you!')

        assert read_file(temp_fn) == ['test line 1', ' test line 2', 
            'test line 3 ', 'thank you!']

        os.remove(temp_fn) # cleanup

    def test_clean_data(self):
        data = ['24 H M V0.0:17 V7.5:15 NULL', '2 B F Z:78 4:7 F2:37 4.0:74']
        icd9_descript_dict = {'V0.0':'P', 'V7.5':'R', 'Z':'O', 
            '4':'V', 'F2':'E', '4.0':'R', '.z':'B'}

        X, y = clean_data(data, icd9_descript_dict=icd9_descript_dict, 
            no_onset_age=True)
        X = [' '.join(f) for f in X] # stringify sublists for easier comparison
        assert X == ['age_24 gender_M V7.5 V0.0', 
            'age_2 gender_F 4 F2 4.0 Z']
        assert y == ['H', 'B']

        X, y = clean_data(data, icd9_descript_dict=icd9_descript_dict, 
            no_onset_age=False)
        X = [' '.join(f) for f in X] # stringify sublists for easier comparison
        assert X == ['age_24 gender_M V7.5:15 V0.0:17', 
            'age_2 gender_F 4:7 F2:37 4.0:74 Z:78']
        assert y == ['H', 'B']

    def test_get_dicts(self):
        X = [['Matthew', '5:5', 'Matthew', 'John', '5:5'], ['5:5'], ['5:5']]
        y = ['A', 'A', 'B']

        feat_idx_dict, idx_feat_dict, class_idx_dict, idx_class_dict = get_dicts(X, y)
        assert feat_idx_dict == {'5:5': 0, 'Matthew': 1, 'John': 2}
        assert idx_feat_dict == {0: '5:5', 1: 'Matthew', 2: 'John'}
        assert class_idx_dict == {'A': 0, 'B': 1}
        assert idx_class_dict == {0: 'A', 1: 'B'}

    def test_encode(self):
        X = [['Matthew', '5:5', 'Matthew', 'John', '5:5'], ['5:5'], ['5:5']]
        y = ['A', 'A', 'B']
        feat_idx_dict = {'5:5': 0, 'Matthew': 1, 'John': 2}
        class_idx_dict = {'A': 0, 'B': 1, 'C': 2}

        X_out, y_out = encode(X, y, feat_idx_dict, class_idx_dict)
        assert X_out == [[1, 0, 1, 2, 0], [0], [0]]
        assert y_out == [0, 0, 1]

    def test_get_k_fold_partition(self):
        X = []
        y = []
        nb_samples = 100
        for i in range(0, nb_samples):
            features = [np.random.randint(11, 50), np.random.randint(11, 50), \
                np.random.randint(11, 50)]
            target = np.random.randint(0, 10)
            X.append(features)
            y.append(target)

        assert len(X) == nb_samples
        assert len(y) == nb_samples

        k = 10
        perm_indices = np.random.permutation(nb_samples)
        test_partitions = []
        for k_idx in range(0, k):
            partition = get_k_fold_partition(X, y, k_idx, k, perm_indices)
            
            # check sum of lengths
            assert len(partition['X_train']) + len(partition['X_val']) + \
                len(partition['X_test']) == nb_samples
            assert len(partition['y_train']) + len(partition['y_val']) + \
                len(partition['y_test']) == nb_samples

            # check identity of collection
            assert sorted(partition['X_train'] + partition['X_val'] + 
                partition['X_test']) == sorted(X)
            assert sorted(partition['y_train'] + partition['y_val'] + 
                partition['y_test']) == sorted(y)

            test_partitions.extend(partition['X_test'])

        assert len(test_partitions) == nb_samples
        assert sorted(test_partitions) == sorted(X)

    def test_get_icd9_descript_dict(self):
        temp_fn = 'ut_icd9_descript.temp'
        with open(temp_fn, 'w+') as f:
            f.write('header_should_not_process\n')
            f.write('0\tA\tB\tC\tD\tE\tF\tG\n')
            f.write('H.5\tI\tJ\tK\tL\tM\tN\tO\n')
            f.write('P.1\tQ\tR\tS\tT\tU\tV\tW\n')

        expected_dict = {'0':'A (category)', 'H.5':'I', 'P.1':'Q'}
        assert get_icd9_descript_dict(temp_fn) == expected_dict

        os.remove(temp_fn) # cleanup
