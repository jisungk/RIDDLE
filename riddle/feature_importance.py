"""
feature_importance.py

Computes feature contribution scores via DeepLIFT (Shrikumar et al., 2016).
Determines most important features via paired t-test with adjustment
for multiple comparisons (Bonferroni correction) using said scores; compares
score vectors between different classes for a fixed feature. 

Requires:   NumPy, SciPy, DeepLIFT (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

import sys
import math
import copy
from collections import OrderedDict

import numpy as np
from scipy import stats

from .models import chunks
from .summary import Summary

# ----------------------------- DEEPLIFT IMPORT ------------------------------ #

# import deeplift, configure path if not already installed
from os.path import abspath, dirname
DEEPLIFT_DIR = dirname(dirname(abspath(__file__))) + '/deeplift'
sys.path.append(DEEPLIFT_DIR)

from deeplift.conversion import keras_conversion as kc
from deeplift.blobs import NonlinearMxtsMode

# -------------------------------- SETTINGS ---------------------------------- #

# how to handle floating pt errs
np.seterr(divide='ignore', over='raise', under='raise')

# ----------------------------- HELPER CLASSES ------------------------------- #

'''
* Helper class to store summary information about feature contribution scores.
'''
class FeatureImportanceSummary(Summary):
    def __init__(self, ordered_dict):
        return super(FeatureImportanceSummary, self).__init__(ordered_dict)

# ---------------------------- HELPER FUNCTIONS ------------------------------ #

''' 
* A generator which computes feature contribution scores via 
  deepLIFT in a batchwise fashion (due to memory concerns). 
* Expects:
    - model = trained Keras model
    - X_test = feature data of the test set as a list of list of feature indices
        e.g., [[0, 1, 2], [2, 3, 4]]
    - process_X_data_func = function to process feature data
    - nb_features = number of features
    - nb_classes = number of classes
    - batch_size = number of samples per batch
    - process_X_data_func_args = additional arguments pass to 
      process_X_data_func(); note: X is already passed as the first argument
'''
def deeplift_contribs_generator(model, X_test, process_X_data_func, 
    nb_features, nb_classes, batch_size, process_X_data_func_args={}):
    
    # convert Keras model, and get relevant function
    deeplift_model = kc.convert_sequential_model(model, num_dims=2,
        nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)
    get_deeplift_contribs = \
        deeplift_model.get_target_contribs_func(find_scores_layer_idx=0)

    # yield a 3D array detailing the deeplift contrib scores
    for X in chunks(X_test, batch_size):
        X = process_X_data_func(X, **process_X_data_func_args)
        batch_size = len(X)
        zeros = [0.0] * batch_size # reference data
        all_batch_contribs = np.zeros((nb_classes, batch_size, nb_features))

        for c in range(nb_classes):
            batch_contribs = get_deeplift_contribs(task_idx=c, 
                input_data_list=[X], input_references_list=zeros, batch_size=10, 
                progress_update=None)
            all_batch_contribs[c] = batch_contribs

        yield all_batch_contribs

'''
* Computes sums of differences, D and D^2, from steaming deepLIFT contribution
  score data. It is very inefficient in terms of memory to store entire matrices 
  of contribution scores so paired t-test computations must be done in a
  streaming fashion with running sums of differences (D) and D^2 (D2), 
  motivating the computation of these running sums.
* Expects:
    - generator = deepLIFT contribution score generator which yields 
      (m,n,f)-dimension numpy arrays; `m` is the number of output neurons 
      (classes), `n` is the batch size and `f` is the number of features
    - nb_features = number of features
    - nb_classes = number of classes
* Returns:
    - two 2D numpy arrays representing running sums of differences (D, D^2) from 
      the generator; rows (first index) represent indices of pairs of compared 
      classes, columns (second index) represent feature indices
    - 2D numpy array representing the sum of feature-class contribution scores; 
      rows represent class indices, columns represent feature indices
    - list of duples of class indices that were compared
        e.g., [(0, 1), (0, 2), (1, 2)]
'''
def diff_sums_from_generator(generator, nb_features, nb_classes):
    # find unique pairs
    pairs = [[(i, j) for j in range(i + 1, nb_classes)] \
        for i in range(nb_classes)]
    pairs = [p for sublist in pairs for p in sublist] # flatten
    nb_pairs = len(pairs)

    # array of running sums of differences (D) and D^2 (D2)
    # for each pair (row) for each feature (column)
    running_sums_D = np.zeros((nb_pairs, nb_features))
    running_sums_D2 = np.zeros((nb_pairs, nb_features))
    # array of running sums of contribution scores
    # for each class (row) for each feature (column)
    running_sums_contribs = np.zeros((nb_classes, nb_features))

    # compute running sums for each pair of classes and their D, D2 values,
    # updating these values batch-by-batch
    for batch_contrib_scores in generator:
        for idx in range(nb_classes):
            contribs = batch_contrib_scores[idx]

            # if only 1 row (e.g., vector), do not sum, will sum all elements
            if (contribs.ndim > 1):
                contribs = np.sum(contribs, axis=0)          

            running_sums_contribs[idx] = np.add(running_sums_contribs[idx], 
                contribs)

        for idx, (i, j) in enumerate(pairs):
            D = np.subtract(batch_contrib_scores[i], batch_contrib_scores[j])
            D2 = np.square(D)

            # if only 1 row (e.g., vector), do not sum, will sum all elements
            assert D.ndim == D2.ndim
            if (D.ndim > 1):
                D = np.sum(D, axis=0)
                D2 = np.sum(D2, axis=0)
            
            assert D.shape == (nb_features, )
            assert D2.shape == (nb_features, )
            
            running_sums_D[idx] = np.add(running_sums_D[idx], D)
            running_sums_D2[idx] = np.add(running_sums_D2[idx], D2)

    return running_sums_D, running_sums_D2, running_sums_contribs, pairs

'''
* Performs preparations for determining which features are important
  for discriminating between two classes, computing deepLIFT contribution 
  scores, and sums for differences of these scores between classes
  (to be used for paired t-tests).
* Expects:
    - generator = deepLIFT contribution score generator which yields 
      (m,n,f)-dimension numpy arrays; `m` is the number of output neurons 
      (classes), `n` is the batch size and `f` is the number of features
    - X_test = test set feature data as a list of list of feature indices
        e.g., [[0, 1, 2], [2, 3, 4]]
    - process_X_data_func = function to process feature data
    - nb_features = number of features
    - nb_classes = number of classes
    - batch_size = number of samples per batch
    - process_X_data_func_args = additional arguments pass to 
      process_X_data_func(); note: X is already passed as the first argument
* Returns:
    - two 2D numpy arrays representing sums of differences (D, D^2); 
      rows (first index) represent indices of pairs of compared classes, 
      columns (second index) represent feature indices
    - 2D numpy array representing the sum of feature-class contribution scores; 
      rows represent class indices, columns represent feature indices
    - list of duples of class indices that were compared
        e.g., [(0, 1), (0, 2), (1, 2)]
'''
def get_diff_sums(model, X_test, process_X_data_func, nb_features, nb_classes, 
    batch_size=128, process_X_data_func_args={}):
    dlc_generator = deeplift_contribs_generator(model, X_test, 
        process_X_data_func, nb_features=nb_features, nb_classes=nb_classes,
        batch_size=batch_size, process_X_data_func_args=process_X_data_func_args)

    sums_D, sums_D2, sums_contribs, pairs = \
        diff_sums_from_generator(dlc_generator, nb_features=nb_features, 
            nb_classes=nb_classes)

    return sums_D, sums_D2, sums_contribs, pairs

'''
* Applies Bonferroni adjustment to p-values.
* Expects:
    - p_values = numpy array of p-values
        e.g., np.array([0.05, 0.1])
    - nb_comparisons = number of comparisons made
* Returns:
    - numpy array of adjusted p-values
'''
def bonferroni(p_values, nb_comparisons):
    adjust = np.vectorize(lambda pv, nb_comp: min(1.0, pv * nb_comp))

    adjusted_p_values = adjust(p_values, nb_comparisons)

    assert np.all(adjusted_p_values[~np.isnan(adjusted_p_values)] <= 1.0)
    assert np.all(adjusted_p_values[~np.isnan(adjusted_p_values)] >= 0.0)

    return adjusted_p_values

'''
* Performs paired t-test with sums of differences, D and D^2
* Expects:
    - sums_D, sums_D2 = two 2D arrays representing sums of differences 
      (D, D^2); rows (first index) represent indices of pairs of compared 
      classes, columns (second index) represent feature indices
    - pairs = list of duples of class indices that were compared
        e.g., [(0, 1), (0, 2), (1, 2)]
    - nb_cases = number of samples
* Returns:
    - two 2D arrays of unadjusted T and adjusted p-values; 
      rows (first index) represent indices of pairs of compared classes, 
      columns (second index) represent feature indices
'''
def paired_ttest_with_diff_sums(sums_D, sums_D2, pairs, nb_cases):
    nb_pairs = len(pairs)
    nb_features = len(sums_D[0])

    # compute T for each pair of classes
    unadjusted_t_values = np.empty((nb_pairs, nb_features)) # placeholder

    for idx, (i, j) in enumerate(pairs):
        sum_D = sums_D[idx]
        sum_D2 = sums_D2[idx]

        assert np.all(~np.isnan(sum_D)); assert np.all(~np.isnan(sum_D2))

        N = float(nb_cases)
        N_minus_1 = float(nb_cases - 1)

        # paired t-test formula from sums of differences
        t = sum_D / np.sqrt((sum_D2 * N - sum_D * sum_D) / N_minus_1)

        unadjusted_t_values[idx] = t

    dof = nb_cases - 1 # degrees of freedom

    # compute two-sided p-value, e.g., Pr(abs(t)> tt)
    unadjusted_p_values = stats.t.sf(np.abs(unadjusted_t_values), dof) * 2
    assert unadjusted_p_values.shape == (nb_pairs, nb_features)

    # apply Bonferroni adjustment to p-values (multiply by # comparisons)
    nb_comparisons = len(pairs) * nb_features
    p_values = bonferroni(unadjusted_p_values, nb_comparisons=nb_comparisons)
    assert p_values.shape == (nb_pairs, nb_features)

    return unadjusted_t_values, p_values

'''
* Gets two flattened lists of unadjusted T, and adjusted p values. Flattens
  matrices so that scores corresponding to the same pair of compared classes
  are continuous, e.g., [f0_p0, f1_p0, f2_p0, f0_p1, f1_p1, f2_p1, ...].
* Expects:
    - unadjusted_t_values, p_values = two 2D arrays of unadjusted T and 
      adjusted p-values; rows (first index) represent indices of pairs of 
      compared classes, columns (second index) represent feature indices
* Returns:
    - list of unadjusted T values
    - list of adjusted p-values
'''
def get_list_signif_scores(unadjusted_t_values, p_values):
    nb_pairs = unadjusted_t_values.shape[0]
    nb_features = unadjusted_t_values.shape[1]

    # flatten nested lists ('C' for row-major, e.g. C style)
    # e.g., np.array([[1, 2, 3], [4, 5, 6]]) => np.array([1, 2, 3, 4, 5, 6])
    # e.g., corresponds to concatenated rows [row0_col0, row1_col0, row2_col0,
    #       row0_col1, row1_col1, row2_col1, row0_col2, row1_col2, row2_col2]
    flat_utv = unadjusted_t_values.flatten('C')
    flat_pv = p_values.flatten('C')

    assert flat_utv.shape == (nb_features * nb_pairs, )
    assert flat_pv.shape == (nb_features * nb_pairs, )

    return flat_utv.tolist(), flat_pv.tolist()

'''
* Gets a list of pairs, corresponding to the flattened list of significance
  scores. 
* Expects:
    - pairs = list of duples of class indices that were compared
        e.g., [(0, 1), (0, 2), (1, 2)]
    - idx_class_dict = dictionary mapping class indices to class names
    - nb_features = number of features
* Returns:
    - list of duples of strings corresponding to class names
        e.g., [('O', 'H'), ('O', 'H'), ('O', 'H'), ('O', 'B'), ('O', 'B'), 
               ('O', 'B'), ('H', 'B'), ('H', 'B'), ('H', 'B')]
'''
def get_list_pairs(pairs, idx_class_dict, nb_features):
    list_pairs = [[p] * nb_features for p in pairs]
    list_pairs = [p for sublist in list_pairs for p in sublist] # flatten
    list_pairs = [[idx_class_dict[p[0]], idx_class_dict[p[1]]] \
        for p in list_pairs] # lookup class
    return list_pairs

'''
* Gets a list of feature names, corresponding to the flattened list of 
  significance scores.
* Expects:
    - idx_feat_dict = dictionary mapping feature indices to feature names
    - nb_pairs = number of pairs of classes that were compared
* Returns:
    - list of string features
        e.g., ['age_34', 'age_32', 'V120', 'age_34', 'age_32', 'V120'] 
               for nb_pairs = 2
'''
def get_list_feat_names(idx_feat_dict, nb_pairs):
    nb_features = len(idx_feat_dict)
    return [idx_feat_dict[feat_idx] for feat_idx in range(nb_features)] \
        * nb_pairs

'''
* Gets a list of feature descriptions, corresponding to the flattened list of 
  significance scores.
* Expects:
    - list_feat_names = list of feature names
    - icd9_descript_dict = dictionary mapping ICD9 codes to their descriptions
* Returns:
    - list of string feature descriptions
'''
def get_list_feat_descripts(list_feat_names, icd9_descript_dict):
    # returns the description for a feature; expects the string feature name 
    def get_descript(feat, icd9_descript_dict):
        if feat[:6] == 'gender': return 'gender'
        elif feat[:3] == 'age': return 'age on record'
        else: return icd9_descript_dict[feat]

        raise ValueError('`{}` not age/gender; not found in icd9_descript_dict'.
            format(feat))

    descripts = [get_descript(f, icd9_descript_dict=icd9_descript_dict) \
        for f in list_feat_names]

    return descripts

# ---------------------------- PUBLIC FUNCTIONS ------------------------------ #

'''
* Gets a Summary object describing information on the importance of specific
  features.
* Expects:
    - sums_D, sums_D2 = two 2D arrays representing sums of differences 
      (D, D^2); rows (first index) represent indices of pairs of compared 
      classes, columns (second index) represent feature indices
        e.g., [(0, 1), (0, 2), (1, 2)]
    - idx_feat_dict = dictionary mapping feature indices to feature names
    - idx_class_dict = dictionary mapping class indices to class names
    - icd9_descript_dict = dictionary mapping ICD9 codes to their descriptions
    - pairs = list of duples of class indices that were compared
        e.g., [(0, 1), (0, 2), (1, 2)]
    - nb_cases = number of samples
* Returns:
    - Summary object
'''
def summarize_feature_importance(sums_D, sums_D2, idx_feat_dict, idx_class_dict, 
    icd9_descript_dict, pairs, nb_cases):
    nb_features = len(idx_feat_dict)
    nb_pairs = len(pairs)

    unadjusted_t_values, p_values = paired_ttest_with_diff_sums(sums_D, 
            sums_D2, pairs=pairs, nb_cases=nb_cases)

    list_unadjusted_t, list_p = get_list_signif_scores(unadjusted_t_values, 
        p_values)
    list_pairs = get_list_pairs(pairs, idx_class_dict=idx_class_dict, 
        nb_features=nb_features)

    list_feat_names = get_list_feat_names(idx_feat_dict, nb_pairs)
    list_feat_descripts = get_list_feat_descripts(list_feat_names, 
        icd9_descript_dict=icd9_descript_dict)

    summary = FeatureImportanceSummary(OrderedDict([('feat', list_feat_names), 
        ('descript', list_feat_descripts), ('pair', list_pairs), 
        ('unadjusted_t', list_unadjusted_t), ('p', list_p)]))

    return summary