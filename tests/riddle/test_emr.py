"""Unit test(s) for emr.py"""

import pytest

import os

import numpy as np

from riddle import emr


class TestEMR():

    def test__read_file(self):
        temp_fn = 'ut_emr_data.temp'

        with open(temp_fn, 'w+') as f:
            f.write('test line 1\n test line 2\ntest line 3 \nthank you!')

        expected = ['test line 1', ' test line 2',
                    'test line 3 ', 'thank you!']
        assert emr._read_file(temp_fn) == expected

        os.remove(temp_fn)  # cleanup

    def test__clean_data(self):
        data = ['24 H M V0.0:17 V7.5:15 NULL', '2 B F Z:78 4:7 F2:37 4.0:74']
        icd9_descript_dict = {'V0.0': 'P', 'V7.5': 'R', 'Z': 'O', '4': 'V',
                              'F2': 'E', '4.0': 'R', '.z': 'B'}

        x, y = emr._clean_data(data, icd9_descript_dict=icd9_descript_dict,
                               no_onset_age=True)
        # stringify sublists for easier comparison
        x = [' '.join(f) for f in x]
        expected = ['age_24 gender_M V7.5 V0.0', 'age_2 gender_F 4 F2 4.0 Z']
        assert x == expected
        assert y == ['H', 'B']

        x, y = emr._clean_data(data, icd9_descript_dict=icd9_descript_dict,
                               no_onset_age=False)
        # stringify sublists for easier comparison
        x = [' '.join(f) for f in x]
        expected = ['age_24 gender_M V7.5:15 V0.0:17',
                    'age_2 gender_F 4:7 F2:37 4.0:74 Z:78']
        assert x == expected
        assert y == ['H', 'B']

    def test__get_dicts(self):
        x = [['Matthew', '5:5', 'Matthew', 'John', '5:5'], ['5:5'], ['5:5']]
        y = ['A', 'A', 'B']

        feat_idx_dict, idx_feat_dict, class_idx_dict, idx_class_dict = \
            emr._get_dicts(x, y)
        assert feat_idx_dict == {'5:5': 0, 'Matthew': 1, 'John': 2}
        assert idx_feat_dict == {0: '5:5', 1: 'Matthew', 2: 'John'}
        assert class_idx_dict == {'A': 0, 'B': 1}
        assert idx_class_dict == {0: 'A', 1: 'B'}

    def test_encode(self):
        x = [['Matthew', '5:5', 'Matthew', 'John', '5:5'], ['5:5'], ['5:5']]
        y = ['A', 'A', 'B']
        feat_idx_dict = {'5:5': 0, 'Matthew': 1, 'John': 2}
        class_idx_dict = {'A': 0, 'B': 1, 'C': 2}

        x_out, y_out = emr._encode(x, y, feat_idx_dict, class_idx_dict)
        assert x_out == [[1, 0, 1, 2, 0], [0], [0]]
        assert y_out == [0, 0, 1]

    def test_get_k_fold_partition(self):
        num_sample = 100
        x, y = [], []
        for i in range(0, num_sample):
            features = [np.random.randint(11, 50), np.random.randint(11, 50),
                        np.random.randint(11, 50)]
            target = np.random.randint(0, 10)
            x.append(features)
            y.append(target)

        assert len(x) == num_sample
        assert len(y) == num_sample

        k = 10
        perm_indices = np.random.permutation(num_sample)
        test_partitions = []
        for k_idx in range(0, k):
            x_train, y_train, x_val, y_val, x_test, y_test = \
                emr.get_k_fold_partition(x, y, k_idx, k, perm_indices)
            # check sum of lengths
            assert num_sample == len(x_train) + len(x_val) + len(x_test)
            assert num_sample == len(y_train) + len(y_val) + len(y_test)

            # check values are equivalent
            assert sorted(x) == sorted(x_train + x_val + x_test)
            assert sorted(y) == sorted(y_train + y_val + y_test)

            test_partitions.extend(x_test)

        assert len(test_partitions) == num_sample
        assert sorted(test_partitions) == sorted(x)  # check full test coverage

    def test_get_icd9_descript_dict(self):
        temp_fn = 'ut_icd9_descript.temp'
        with open(temp_fn, 'w+') as f:
            f.write('header_should_not_process\n')
            f.write('0\tA\tB\tC\tD\tE\tF\tG\n')
            f.write('H.5\tI\tJ\tK\tL\tM\tN\tO\n')
            f.write('P.1\tQ\tR\tS\tT\tU\tV\tW\n')

        expected_dict = {'0': 'A (category)', 'H.5': 'I', 'P.1': 'Q'}
        assert emr.get_icd9_descript_dict(temp_fn) == expected_dict

        os.remove(temp_fn)  # cleanup
