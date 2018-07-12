"""Unit test(s) for ordering.py"""

import pytest

import os
import shutil
from collections import OrderedDict
import copy
import pickle

import numpy as np

from riddle import ordering

PRECISION = 4


class TestSummary:

    @pytest.fixture(scope='module')
    def summary(self):
        list_feat = ['John', 'James', 'Christine']
        list_descript = ['kind', 'caring', 'righteous']
        list_contrib_ordering = [
            [['A', 5.0], ['B', 3.0], ['C', -1.0], ['D', -5.0]],
            [['D', 100.0], ['C', 99.0], ['B', 98.0], ['A', 97.0]],
            [['C', -4.0], ['D', -5.0], ['B', -7.0], ['A', -132.0]]
        ]
        list_freq_ordering = [
            [['A', 300.0], ['B', 200.0], ['C', 100.0], ['D', 0.0]],
            [['D', 5.0], ['C', 5.0], ['B', 5.0], ['A', 5.0]],
            [['C', 2.0], ['D', 23.0], ['B', 34.0], ['A', 45.0]]
        ]

        od = OrderedDict([('feat', list_feat), ('descript', list_descript),
                          ('contrib_ordering', list_contrib_ordering),
                          ('freq_ordering', list_freq_ordering)])
        return ordering.OrderingSummary(od)

    def test_sort(self, summary):
        # TODO(jisungk): update when sort() is implemented.
        return

        # not sorted this line; will be
        sorted_summary = copy.deepcopy(summary)
        sorted_summary.sort()
        sorted_od = sorted_summary.od

        assert sorted_od['feat'] == ['Christine', 'John', 'James']
        assert sorted_od['descript'] == ['righteous', 'kind', 'caring']
        assert sorted_od['contrib_ordering'] == [
            [['C', -4.0], ['D', -5.0], ['B', -7.0], ['A', -132.0]],
            [['A', 5.0], ['B', 3.0], ['C', -1.0], ['D', -5.0]],
            [['D', 100.0], ['C', 99.0], ['B', 98.0], ['A', 97.0]]
        ]
        assert sorted_od['freq_ordering'] == [
            [['C', 2.0], ['D', 23.0], ['B', 34.0], ['A', 45.0]],
            [['A', 300.0], ['B', 200.0], ['C', 100.0], ['D', 0.0]],
            [['D', 5.0], ['C', 5.0], ['B', 5.0], ['A', 5.0]]
        ]

    def test_save(self, summary):
        def stringify_pair(p):
            return p[0] + ' (' + str(round(p[1], PRECISION)) + ')'

        out_directory = 'tests/riddle/temp'
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        summary.save(out_directory)

        with open(out_directory + '/orderings_ordered_dict.pkl', 'r') as f:
            saved_od = pickle.load(f)
            assert summary.od.items() == saved_od.items()

        with open(out_directory + '/orderings.txt', 'r') as f:
            lines = f.read().splitlines()

        assert lines[0] == '\t'.join(summary.od.keys())

        list_feat = summary.od['feat']
        list_descript = summary.od['descript']

        list_contrib_ordering = [
            ' > '.join([stringify_pair(p) for p in o])
            for o in summary.od['contrib_ordering']
        ]
        list_freq_ordering = [
            ' > '.join([stringify_pair(p) for p in o])
            for o in summary.od['freq_ordering']
        ]

        for row_idx in range(1, len(summary.od.values()[0])):
            expected = '\t'.join(
                [list_feat[row_idx - 1],
                 list_descript[row_idx - 1],
                 list_contrib_ordering[row_idx - 1],
                 list_freq_ordering[row_idx - 1]]
            )
            assert lines[row_idx] == expected

        shutil.rmtree(out_directory)

    def test_save_individual_tables(self, summary):
        out_directory = 'tests/riddle/temp'
        if not os.path.exists(out_directory):
            os.makedirs(out_directory)
        summary.save(out_directory)

        idx_class_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        sorted_idx_class = sorted(idx_class_dict.items(), key=lambda x: x[0])

        summary.save_individual_tables(idx_class_dict, out_directory)

        ordering_keys = [key for key in summary.od.keys() if 'ordering' in key]
        for key in ordering_keys:
            features = summary.od['feat']
            curr_table_data = summary.od[key]
            with open(out_directory + '/' + key + '_table.txt', 'r') as f:
                lines = f.read().splitlines()
                assert lines[0] == 'feat\tA\tB\tC\tD'
                for row_idx in range(1, len(summary.od.values()[0])):
                    def search_score(c, list_pairs):
                        c_index = [cl for cl, score in list_pairs].index(c)
                        return list_pairs[c_index][1]

                    feat = features[row_idx - 1]
                    list_pairs = curr_table_data[row_idx - 1]
                    sorted_scores = [str(search_score(c, list_pairs))
                                     for idx, c in sorted_idx_class]
                    expected_line = '\t'.join([feat] + sorted_scores)
                    assert lines[row_idx] == expected_line

        shutil.rmtree(out_directory)


def test__compute_orderings():
    # orderings by contribution score
    sums_contribs = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [-5, -5, -5, -5]])
    contrib_orderings = ordering._compute_orderings(sums_contribs)
    expected_contrib_orderings = [
        [(1, 4), (0, 1), (2, -5)],
        [(1, 3), (0, 2), (2, -5)],
        [(0, 3), (1, 2), (2, -5)],
        [(0, 4), (1, 1), (2, -5)]
    ]
    assert np.all(contrib_orderings == expected_contrib_orderings)

    # orderings by frequencies
    class_feat_freq_table = np.array([[5, 3, 4], [6, 1, 3], [7, 0, 2],
                                      [9, 4, 1]])
    freq_orderings = ordering._compute_orderings(class_feat_freq_table)
    expected_freq_orderings = [
        [(3, 9), (2, 7), (1, 6), (0, 5)],
        [(3, 4), (0, 3), (1, 1), (2, 0)],
        [(0, 4), (1, 3), (2, 2), (3, 1)]
    ]
    assert np.all(freq_orderings == expected_freq_orderings)

    # orderings by frequency proportions
    class_feat_prop_table = ordering._compute_prop_table(class_feat_freq_table)
    prop_orderings = ordering._compute_orderings(class_feat_prop_table)
    expected_prop_orderings = [
        [(2, 7. / 9),  (3, 9. / 14), (1, 6. / 10), (0, 5. / 12)],
        [(3, 4. / 14), (0, 3. / 12), (1, 1. / 10), (2, 0.)],
        [(0, 4. / 12), (1, 3. / 10), (2, 2. / 9),  (3, 1. / 14)]
    ]
    assert np.all(prop_orderings == expected_prop_orderings)


def test__compute_prop_table():
    class_feat_freq_table = np.array([[5, 3, 4], [6, 1, 3], [7, 0, 2],
                                      [9, 4, 1]])
    expected_class_feat_prop_table = [
        [5. / 12, 3. / 12, 4. / 12],
        [6. / 10, 1. / 10, 3. / 10],
        [7. / 9,  0. / 9,  2. / 9],
        [9. / 14, 4. / 14, 1. / 14]
    ]
    class_feat_prop_table = ordering._compute_prop_table(class_feat_freq_table)
    assert np.all(class_feat_prop_table == expected_class_feat_prop_table)


def test_decode_orderings():
    raw_orderings = [
        [[0, 0.5], [1, 1.5], [2, 2.5], [3, 3.5]],
        [[3, 3], [2, 2], [1, 1], [0, 0]]
    ]
    idx_class_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

    decoded_orderings = ordering._decode_orderings(raw_orderings,
                                                   idx_class_dict)
    expected_decoded_orderings = [
        [('A', 0.5), ('B', 1.5), ('C', 2.5), ('D', 3.5)],
        [('D', 3), ('C', 2), ('B', 1), ('A', 0)]
    ]
    assert decoded_orderings == expected_decoded_orderings
