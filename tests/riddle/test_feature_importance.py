"""Unit test(s) for feature_importance.py"""

import pytest

import os
import pickle

import numpy as np
from scipy.stats import ttest_rel

from riddle import feature_importance
from riddle.models import MLP

EPSILON = 1e-5  # tolerance


# helper function to remove nan
def del_nans(a):
    b = np.array(a)
    return b[~np.isnan(b)]


@pytest.fixture(scope='module')
def data():
    x_test_fn, model_fn = 'ut_x_test.pkl', 'ut_model.h5'

    def find(name):
        path = os.getcwd()
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)

        raise ValueError('file `{}` not found in path `{}`'.format(name, path))

    x_test_path = find(x_test_fn)
    model_path = find(model_fn)

    with open(x_test_path, 'r') as f:
        x_test = pickle.load(f)

    return model_path, x_test


def test__deeplift_contribs_generator(data):
    model_fn, x_test = data

    num_feature, num_class, num_sample = 1717, 4, 20
    batch_size = 5

    temp_model = MLP(num_feature=num_feature, num_class=num_class)
    dlc_gen = feature_importance._deeplift_contribs_generator(
        model_fn,
        x_test,
        process_x_func=temp_model.process_x,
        num_feature=num_feature,
        num_class=num_class,
        batch_size=batch_size)

    for idx, dlc in enumerate(dlc_gen):
        assert len(dlc) == num_class
        for d in dlc:
            if idx <= num_sample / batch_size:
                assert d.shape == (batch_size, num_feature)


def test__diff_sums_from_generator():
    def test_gen_1():  # generator with 1 row per batch
        yield [np.asarray([0.4, 0.1]),
               np.asarray([0.5, 0.7]),
               np.asarray([12, 12])]
        yield [np.asarray([5, 4]),
               np.asarray([3, 2]),
               np.asarray([1, 9])]

    def test_gen_2():  # generator with 2 rows per batch
        yield [np.asarray([[0.4, 0.1], [0.4, 0.1]]),
               np.asarray([[0.5, 0.7], [0.5, 0.7]]),
               np.asarray([[12, 12], [12, 12]])]
        yield [np.asarray([[5, 4], [5, 4]]),
               np.asarray([[3, 2], [3, 2]]),
               np.asarray([[1, 9], [1, 9]])]

    num_feature = 2
    num_class = 3

    running_sums_D, running_sums_D2, running_sums_contribs, pairs = \
        feature_importance._diff_sums_from_generator(
            test_gen_1(),
            num_feature=num_feature,
            num_class=num_class)

    assert pairs == [(0, 1), (0, 2), (1, 2)]

    expected = [np.asarray([1.9, 1.4]),
                np.asarray([-7.6, -16.9]),
                np.asarray([-9.5, -18.3])]
    assert np.all(np.equal(running_sums_D, expected))

    expected = [np.asarray([4.01, 4.36]),
                np.asarray([150.56, 166.61]),
                np.asarray([136.25, 176.69])]
    assert np.all(np.equal(running_sums_D2, expected))

    expected = [np.asarray([5.4, 4.1]),
                np.asarray([3.5, 2.7]),
                np.asarray([13.0, 21.0])]
    assert np.all(np.equal(running_sums_contribs, expected))

    running_sums_D, running_sums_D2, running_sums_contribs, pairs = \
        feature_importance._diff_sums_from_generator(
            test_gen_2(),
            num_feature=num_feature,
            num_class=num_class)

    assert pairs == [(0, 1), (0, 2), (1, 2)]

    expected = [np.asarray([3.8, 2.8]),
                np.asarray([-15.2, -33.8]),
                np.asarray([-19.0, -36.6])]
    assert np.all(np.equal(running_sums_D, expected))

    expected = [np.asarray([8.02, 8.72]),
                np.asarray([301.12, 333.22]),
                np.asarray([272.50, 353.38])]
    assert np.all(np.equal(running_sums_D2, expected))

    expected = [np.asarray([10.8, 8.2]),
                np.asarray([7.0, 5.4]),
                np.asarray([26.0, 42.0])]
    assert np.all(np.equal(running_sums_contribs, expected))


def test__bonferroni():
    assert feature_importance._bonferroni([0.10], 10) == [1.0]
    assert feature_importance._bonferroni([0.95], 5) == [1.0]
    assert feature_importance._bonferroni([0.10], 5) == [0.5]
    assert feature_importance._bonferroni([0.01], 3) == [0.03]


def test__paired_ttest_with_diff_sums(data):
    model_fn, x_test = data

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    num_pair = len(pairs)

    num_feature, num_class, num_sample = 1717, 4, 20
    batch_size = 5
    temp_model = MLP(num_feature=num_feature, num_class=num_class)
    dlc_gen = feature_importance._deeplift_contribs_generator(
        model_fn,
        x_test,
        process_x_func=temp_model.process_x,
        num_feature=num_feature,
        num_class=num_class,
        batch_size=batch_size)

    sums_D, sums_D2, sums_contribs, pairs = \
        feature_importance._diff_sums_from_generator(
            dlc_gen,
            num_feature=num_feature,
            num_class=num_class)

    unadjusted_t_values, p_values = \
        feature_importance._paired_ttest_with_diff_sums(
            sums_D,
            sums_D2,
            pairs=pairs,
            num_sample=num_sample)

    assert unadjusted_t_values.shape == (num_pair, num_feature)
    assert p_values.shape == (num_pair, num_feature)

    # force only 1 batch with abnormally high batch_size parameter
    alt_dlc_gen = feature_importance._deeplift_contribs_generator(
        model_fn,
        x_test,
        process_x_func=temp_model.process_x,
        num_feature=num_feature,
        num_class=num_class,
        batch_size=109971161161043253 % 8085)

    # non-streaming paired t-test implementation... fails with larger
    # datasets due to large matrix sizes (e.g., memory overflow), but
    # works as an alternative implementation for a tiny unit testing dataset
    alt_t_values, alt_p_values = [], []
    for idx, contribs in enumerate(alt_dlc_gen):
        assert not idx  # check only 1 batch (idx == 0)
        for i, j in pairs:
            curr_t_values = np.zeros((num_feature, ))
            curr_p_values = np.zeros((num_feature, ))

            for f in range(num_feature):
                t, p = ttest_rel(contribs[i][:, f], contribs[j][:, f])
                curr_t_values[f] = t
                curr_p_values[f] = p

            alt_t_values.append(curr_t_values)
            alt_p_values.append(curr_p_values)

    for r in range(len(pairs)):
        t = unadjusted_t_values[r]
        alt_t = alt_t_values[r]
        p = p_values[r]  # already bonferroni adjusted
        alt_p = feature_importance._bonferroni(alt_p_values[r],
                                               num_pair * num_feature)

        assert t.shape == alt_t.shape
        assert p.shape == alt_p.shape

        a = del_nans(np.abs(alt_t - t))
        b = a < EPSILON
        print(a)
        print('\n' * 3)
        print(b)
        print('\n' * 3)
        print(a[~b])
        print('\n' * 3)

        assert np.all(del_nans(np.abs(alt_t - t)) < EPSILON)
        assert np.all(del_nans(np.abs(alt_p - p)) < EPSILON)


def test__get_list_signif_scores(data):
    model_fn, x_test = data

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    num_pair = len(pairs)

    num_feature, num_class, num_sample = 1717, 4, 20
    batch_size = 5
    temp_model = MLP(num_feature=num_feature, num_class=num_class)
    dlc_gen = feature_importance._deeplift_contribs_generator(
        model_fn,
        x_test,
        process_x_func=temp_model.process_x,
        num_feature=num_feature,
        num_class=num_class,
        batch_size=batch_size)

    sums_D, sums_D2, sums_contribs, pairs = \
        feature_importance._diff_sums_from_generator(
            dlc_gen,
            num_feature=num_feature,
            num_class=num_class)

    unadjusted_t_values, p_values = \
        feature_importance._paired_ttest_with_diff_sums(
            sums_D,
            sums_D2,
            pairs=pairs,
            num_sample=num_sample)
    list_unadjusted_t, list_p = feature_importance._get_list_signif_scores(
        unadjusted_t_values, p_values)

    alt_list_unadjusted_t, alt_list_p = [], []
    for r in range(num_pair):
        alt_list_unadjusted_t.extend(unadjusted_t_values[r])
        alt_list_p.extend(p_values[r])

    list_unadjusted_t = np.array(list_unadjusted_t)
    alt_list_unadjusted_t = np.array(alt_list_unadjusted_t)
    list_p = np.array(list_p)
    alt_list_p = np.array(alt_list_p)

    delta = list_unadjusted_t - alt_list_unadjusted_t
    assert np.all(del_nans(np.abs(delta)) < EPSILON)

    delta = list_p - alt_list_p
    assert np.all(del_nans(np.abs(delta)) < EPSILON)


def test__get_list_pairs():
    pairs = [[1, 2], [5, 6], [1, 5]]
    idx_class_dict = {1: 'A', 2: 'B', 5: 'C', 6: 'D'}
    num_feature = 3

    list_pairs = feature_importance._get_list_pairs(pairs, idx_class_dict,
                                                    num_feature)

    expected_list_pairs = [
        ['A', 'B'], ['A', 'B'], ['A', 'B'], ['C', 'D'],
        ['C', 'D'], ['C', 'D'], ['A', 'C'], ['A', 'C'], ['A', 'C']
    ]
    assert list_pairs == expected_list_pairs


def test__get_list_feat_names():
    idx_feat_dict = {0: 'English', 1: 'Standard', 2: 'Version'}
    num_pair = 3
    list_feat_names = feature_importance._get_list_feat_names(idx_feat_dict,
                                                              num_pair)
    expected_list_feat_names = [
        'English', 'Standard', 'Version', 'English', 'Standard', 'Version',
        'English', 'Standard', 'Version'
    ]
    assert list_feat_names == expected_list_feat_names


def test__get_list_feat_descripts():
    list_feat_names = [
        'featA', 'featB', 'gender_M', 'age_72', 'featA', 'featB', 'gender_M',
        'age_72'
    ]
    icd9_descript_dict = {'featA': 'descriptionA', 'featB': 'descriptionB'}

    list_feat_descripts = feature_importance._get_list_feat_descripts(
        list_feat_names, icd9_descript_dict)
    expected_list_feat_descripts = [
        'descriptionA', 'descriptionB', 'gender', 'age on record',
        'descriptionA', 'descriptionB', 'gender', 'age on record'
    ]

    assert list_feat_descripts == expected_list_feat_descripts
