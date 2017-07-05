"""
test_feature_importance.py

Unit test(s) for the `feature_importance.py` module.

Requires:   pytest, SciPy, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

import pytest

import sys; sys.dont_write_bytecode = True
import os
import pickle

import numpy as np
from scipy.stats import ttest_rel

from riddle.feature_importance import *
from riddle.models import load_model
from riddle.models.deep_mlp import process_X_data, process_y_data

# tolerance when checking equality of floating point values
epsilon = 1e-6

# helper function to remove nan
def del_nans(a):
    b = np.array(a)
    return b[~np.isnan(b)]

@pytest.fixture(scope='module')
def data():
    X_test_fn, model_fn = 'ut_X_test.pkl', 'ut_model.h5'

    def find(name):
        path = os.getcwd()
        for root, dirs, files in os.walk(path):
            if name in files:
                return os.path.join(root, name)

        raise ValueError('file `{}` not found in path `{}`'.format(name, path))

    X_test_path = find(X_test_fn)
    model_path = find(model_fn)

    with open(X_test_path, 'r') as f: 
        X_test = pickle.load(f) 
    model = load_model(model_path)

    return model, X_test

def test_deeplift_contribs_generator(data):
    model, X_test = data

    nb_features, nb_classes, nb_cases = 1717, 4, 20
    batch_size = 5
    process_X_data_func_args = {'nb_features': nb_features}

    dlc_gen = deeplift_contribs_generator(model, X_test, 
        process_X_data_func=process_X_data, nb_features=nb_features, 
        nb_classes=nb_classes, batch_size=batch_size,
        process_X_data_func_args=process_X_data_func_args)

    for idx, dlc in enumerate(dlc_gen):
        assert len(dlc) == nb_classes
        for d in dlc:
            if idx < nb_cases / batch_size:
                assert d.shape == (batch_size, nb_features)

def test_diff_sums_from_generator():
    nb_features, nb_classes = 2, 3

    def test_gen_1(): # 1 row per batch
        yield [np.asarray([0.4, 0.1]), np.asarray([0.5, 0.7]), \
            np.asarray([12, 12])]
        yield [np.asarray([5, 4]), np.asarray([3, 2]), np.asarray([1, 9])]

    running_sums_D, running_sums_D2, running_sums_contribs, pairs = \
        diff_sums_from_generator(test_gen_1(), nb_features=nb_features, 
            nb_classes=nb_classes)

    assert pairs == [(0, 1), (0, 2), (1, 2)]
    assert np.all(np.equal(running_sums_D, [np.asarray([1.9, 1.4]), 
        np.asarray([-7.6, -16.9]), np.asarray([-9.5, -18.3])]))
    assert np.all(np.equal(running_sums_D2, [np.asarray([4.01, 4.36]), 
        np.asarray([150.56, 166.61]), np.asarray([136.25, 176.69])]))

    print(running_sums_contribs)

    assert np.all(np.equal(running_sums_contribs, [np.asarray([5.4, 4.1]), 
        np.asarray([3.5, 2.7]), np.asarray([13.0, 21.0])]))

    def test_gen_2(): # 2 rows per batch
        yield [np.asarray([[0.4, 0.1], [0.4, 0.1]]), 
            np.asarray([[0.5, 0.7], [0.5, 0.7]]),
            np.asarray([[12, 12], [12, 12]])]
        yield [np.asarray([[5, 4], [5, 4]]), 
            np.asarray([[3, 2], [3, 2]]), 
            np.asarray([[1, 9], [1, 9]])]

    running_sums_D, running_sums_D2, running_sums_contribs, pairs = \
        diff_sums_from_generator(test_gen_2(), nb_features=nb_features, 
            nb_classes=nb_classes)
        
    assert pairs == [(0, 1), (0, 2), (1, 2)]
    assert np.all(np.equal(running_sums_D, [np.asarray([3.8, 2.8]), 
        np.asarray([-15.2, -33.8]), np.asarray([-19.0, -36.6])]))
    assert np.all(np.equal(running_sums_D2, [np.asarray([8.02, 8.72]), 
        np.asarray([301.12, 333.22]), np.asarray([272.50, 353.38])]))
    assert np.all(np.equal(running_sums_contribs, [np.asarray([10.8, 8.2]), 
        np.asarray([7.0, 5.4]), np.asarray([26.0, 42.0])]))

def test_bonferroni():
    assert bonferroni([0.10], 10) == [1.0]
    assert bonferroni([0.95], 5) == [1.0]
    assert bonferroni([0.1], 5) == [0.5]
    assert bonferroni([0.01], 3) == [0.03]

def test_paired_ttest_with_diff_sums(data):
    model, X_test = data

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    nb_pairs = len(pairs)

    nb_features, nb_classes, nb_cases = 1717, 4, 20
    batch_size = 5
    process_X_data_func_args = {'nb_features': nb_features}

    dlc_gen = deeplift_contribs_generator(model, X_test, 
        process_X_data_func=process_X_data, nb_features=nb_features, 
        nb_classes=nb_classes, batch_size=batch_size,
        process_X_data_func_args=process_X_data_func_args)

    sums_D, sums_D2, sums_contribs, pairs = diff_sums_from_generator(dlc_gen, 
        nb_features=nb_features, nb_classes=nb_classes)

    unadjusted_t_values, p_values = paired_ttest_with_diff_sums(sums_D, 
        sums_D2, pairs=pairs, nb_cases=nb_cases)

    assert unadjusted_t_values.shape == (nb_pairs, nb_features)
    assert p_values.shape == (nb_pairs, nb_features)

    # force only 1 batch with abnormally high batch_size parameter
    alt_dlc_gen = deeplift_contribs_generator(model, X_test, 
        process_X_data_func=process_X_data, nb_features=nb_features, 
        nb_classes=nb_classes, batch_size=109971161161043253 % 8085,
        process_X_data_func_args=process_X_data_func_args)

    # non-streaming paired t-test implementation... fails with larger 
    # datasets due to large matrix sizes (e.g., memory overflow), but
    # works as an alternative implementation for a tiny unit testing dataset
    alt_t_values, alt_p_values = [], []
    for idx, contribs in enumerate(alt_dlc_gen):
        assert not idx # check only 1 batch (idx == 0)
        for i, j in pairs:
            curr_t_values = np.zeros((nb_features, ))
            curr_p_values = np.zeros((nb_features, ))

            for f in range(nb_features):
                t, p = ttest_rel(contribs[i][:, f], contribs[j][:, f])
                curr_t_values[f] = t
                curr_p_values[f] = p

            alt_t_values.append(curr_t_values)
            alt_p_values.append(curr_p_values)

    for r in range(len(pairs)):
        t = unadjusted_t_values[r]
        alt_t = alt_t_values[r]
        p = p_values[r] # already bonferroni adjusted
        alt_p = bonferroni(alt_p_values[r], nb_pairs * nb_features)

        assert t.shape == alt_t.shape
        assert p.shape == alt_p.shape

        assert np.all(del_nans(np.abs(alt_t - t)) < epsilon)
        assert np.all(del_nans(np.abs(alt_p - p)) < epsilon)

def test_get_list_signif_scores(data):
    model, X_test = data

    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    nb_pairs = len(pairs)

    nb_features, nb_classes, nb_cases = 1717, 4, 20
    batch_size = 5
    process_X_data_func_args = {'nb_features': nb_features}

    dlc_gen = deeplift_contribs_generator(model, X_test, 
        process_X_data_func=process_X_data, nb_features=nb_features, 
        nb_classes=nb_classes, batch_size=batch_size, 
        process_X_data_func_args=process_X_data_func_args)

    sums_D, sums_D2, sums_contribs, pairs = diff_sums_from_generator(dlc_gen, 
        nb_features=nb_features, nb_classes=nb_classes)

    unadjusted_t_values, p_values = paired_ttest_with_diff_sums(sums_D, 
        sums_D2, pairs=pairs, nb_cases=nb_cases)
    list_unadjusted_t, list_p = get_list_signif_scores(unadjusted_t_values, 
        p_values)

    alt_list_unadjusted_t, alt_list_p = [], []
    for r in range(nb_pairs):
        alt_list_unadjusted_t.extend(unadjusted_t_values[r])
        alt_list_p.extend(p_values[r])

    list_unadjusted_t = np.array(list_unadjusted_t)
    alt_list_unadjusted_t = np.array(alt_list_unadjusted_t)
    list_p = np.array(list_p)
    alt_list_p = np.array(alt_list_p)

    assert np.all(del_nans(np.abs(list_unadjusted_t - alt_list_unadjusted_t)) 
        < epsilon)
    assert np.all(del_nans(np.abs(list_p - alt_list_p)) < epsilon)

def test_get_list_pairs():
    pairs = [[1, 2], [5, 6], [1, 5]] 
    idx_class_dict = {1: 'A', 2: 'B', 5: 'C', 6:'D'}
    nb_features = 3
    
    list_pairs = get_list_pairs(pairs, idx_class_dict, nb_features)

    expected_list_pairs = [['A', 'B'], ['A', 'B'], ['A', 'B'], ['C', 'D'], \
        ['C', 'D'], ['C', 'D'], ['A', 'C'], ['A', 'C'], ['A', 'C']]
    assert list_pairs == expected_list_pairs

def test_get_list_feat_names():
    idx_feat_dict = {0:'English', 1:'Standard', 2:'Version'}
    nb_pairs = 3
    list_feat_names = get_list_feat_names(idx_feat_dict, nb_pairs)
    expected_list_feat_names = ['English', 'Standard', 'Version', 'English', \
        'Standard', 'Version', 'English','Standard','Version']
    assert list_feat_names == expected_list_feat_names

def test_get_list_feat_descripts():
    list_feat_names = ['featA', 'featB', 'gender_M', 'age_72', 'featA', \
        'featB', 'gender_M', 'age_72']
    icd9_descript_dict = {'featA':'descriptionA', 'featB':'descriptionB'}

    list_feat_descripts = get_list_feat_descripts(list_feat_names, 
        icd9_descript_dict)
    expected_list_feat_descripts = ['descriptionA', 'descriptionB', 'gender', \
        'age on record', 'descriptionA', 'descriptionB', 'gender', \
        'age on record']

    assert list_feat_descripts == expected_list_feat_descripts