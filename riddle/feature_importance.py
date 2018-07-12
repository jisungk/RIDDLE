"""feature_importance.py

Computes feature contribution scores via DeepLIFT (Shrikumar et al., 2016) &
determines most important features via paired t-test with adjustment
for multiple comparisons (Bonferroni correction) using said scores.

Requires:   NumPy, SciPy, DeepLIFT (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

from __future__ import print_function

from collections import OrderedDict
from os.path import abspath
from os.path import dirname
import sys
import time

import numpy as np
from scipy import stats

from .models import chunks
from .summary import Summary

# import deeplift, configure path if not already installed
sys.path.append(dirname(dirname(abspath(__file__))) + '/deeplift')
from deeplift.conversion import kerasapi_conversion as kc
from deeplift.layers import NonlinearMxtsMode

# how to handle floating pt errs
np.seterr(divide='ignore', over='raise', under='raise')


class FeatureImportanceSummary(Summary):
    """Feature importance summary."""

    def __init__(self, sums_D, sums_D2, idx_feat_dict, idx_class_dict,
                 icd9_descript_dict, pairs, num_sample):
        """Initialize feature importance summary.

        Arguments:
            sums_D: np.ndarray, float
                2-D array of sums of differences in DeepLIFT contribution scores
                with shape (num_pair, num_feature); the outer (0) dim represents the
                pair of compared classes, and the inner dim (1) represents the sum
                of differences in scores across features
            sums_D2: np.ndarray, float
                2-D array of sums of squared differences in DeepLIFT contribution
                scores with shape (num_pair, num_feature); the outer (0) dim
                represents the pair of compared classes, and the inner dim (1)
                represents the sum of squared differences in scores across features
            idx_feat_dict: {int: string}
                dictionary mapping feature indices to features
            idx_class_dict: {int: string}
                dictionary mapping class indices to classes
            icd9_descript_dict: {string: string}
                dictionary mapping ICD9 codes to description text
            pairs: [(int, int)]
                list of pairs of classes which were compared during interpretation
            num_sample: int
                number of samples present in the dataset
        """
        num_feature = len(idx_feat_dict)
        num_pair = len(pairs)

        unadjusted_t_values, p_values = _paired_ttest_with_diff_sums(
            sums_D, sums_D2, pairs=pairs, num_sample=num_sample)

        list_unadjusted_t, list_p = _get_list_signif_scores(
            unadjusted_t_values, p_values)
        list_pairs = _get_list_pairs(pairs, idx_class_dict=idx_class_dict,
                                     num_feature=num_feature)

        list_feat_names = _get_list_feat_names(idx_feat_dict, num_pair)
        list_feat_descripts = _get_list_feat_descripts(
            list_feat_names, icd9_descript_dict=icd9_descript_dict)

        super(FeatureImportanceSummary, self).__init__(OrderedDict(
            [('feat', list_feat_names),
             ('descript', list_feat_descripts), ('pair', list_pairs),
             ('unadjusted_t', list_unadjusted_t), ('p', list_p)]))


def get_diff_sums(hdf5_path, x_test, process_x_func, num_feature, num_class,
                  batch_size=1024):
    """Get differences in sums of contribution score values.

    Performs preparations for determining hich features are important
    for discriminating between two classes, computing DeepLIFT contribution
    scores, and sums for differences of these scores between classes
    (to be used for paired t-tests).

    Arguments:
        hdf5_path: str
            path to saved HDF5 Keras Model
        process_x_func: function
            function for vectorizing feature data
        num_feature: int
            number of features present in the dataset
        num_class: int
            number of classes
        batch_size: int
            batch size

    Returns:
        sums_D: np.ndarray, float
            2-D array of sums of differences in DeepLIFT contribution scores
            with shape (num_pair, num_feature); the outer (0) dim represents the
            pair of compared classes, and the inner dim (1) represents the sum
            of differences in scores across features
        sums_D2: np.ndarray, float
            2-D array of sums of squared differences in DeepLIFT contribution
            scores with shape (num_pair, num_feature); the outer (0) dim
            represents the pair of compared classes, and the inner dim (1)
            represents the sum of squared differences in scores across features
        sums: np.ndarray, float
            2-D array of sums of DeepLIFT contribution scores with shape
            (num_class, num_feature); the outer (0) dim represents the pair of
            compared classes, and the inner dim (1) represents the sum of
            differences in scores across features
        pairs: [(int, int)]
            list of pairs of classes which were compared during interpretation
    """
    dlc_generator = _deeplift_contribs_generator(
        hdf5_path, x_test, process_x_func, num_feature=num_feature,
        num_class=num_class, batch_size=batch_size)

    sums_D, sums_D2, sums_contribs, pairs = _diff_sums_from_generator(
        dlc_generator, num_feature=num_feature, num_class=num_class)

    return sums_D, sums_D2, sums_contribs, pairs


def _deeplift_contribs_generator(hdf5_path, x_test, process_x_func,
                                 num_feature, num_class, batch_size):
    """Generator which yields DeepLIFT contribution scores.

    Applies vectorization batch-by-batch to avoid memory overflow.

    Arguments:
        hdf5_path: str
            path to saved HDF5 Keras Model
        process_x_func: function
            function for vectorizing feature data
        num_feature: int
            number of features present in the dataset
        num_class: int
            number of classes
        batch_size: int
            batch size
    """
    # convert Keras model, and get relevant function
    deeplift_model = kc.convert_model_from_saved_files(
        hdf5_path, nonlinear_mxts_mode=NonlinearMxtsMode.RevealCancel)
    # input layer is 0, since we have a softmax layer the target layer is -2
    get_deeplift_contribs = deeplift_model.get_target_contribs_func(
        find_scores_layer_idx=0, target_layer_idx=-2)

    num_batch = int(round(float(len(x_test)) / batch_size))
    # yield a 3D array detailing the DeepLIFT contrib scores
    for batch_idx, x in enumerate(chunks(x_test, batch_size)):
        start = time.time()
        x = process_x_func(x)
        batch_size = len(x)
        zeros = [0.0] * batch_size  # reference data
        all_batch_contribs = np.zeros((num_class, batch_size, num_feature))

        for c in range(num_class):
            batch_contribs = get_deeplift_contribs(
                task_idx=c, input_data_list=[x], input_references_list=zeros,
                batch_size=1024, progress_update=None)
            all_batch_contribs[c] = batch_contribs

        if not batch_idx % 10:
            print('{}/{} in {:.2f} s'.format(batch_idx, num_batch,
                                             time.time() - start))

        yield all_batch_contribs


def _diff_sums_from_generator(generator, num_feature, num_class):
    """Computes sums of DeepLIFT contribution scores from a generator.

    Arguments:
        generator: generator
            generator which yields DeepLIFT contribution scores.
        num_feature: int
            number of features present in the dataset
        num_class: int
            number of classes

    Returns:
        sums_D: np.ndarray, float
            2-D array of sums of differences in DeepLIFT contribution scores
            with shape (num_pair, num_feature); the outer (0) dim represents the
            pair of compared classes, and the inner dim (1) represents the sum
            of differences in scores across features
        sums_D2: np.ndarray, float
            2-D array of sums of squared differences in DeepLIFT contribution
            scores with shape (num_pair, num_feature); the outer (0) dim
            represents the pair of compared classes, and the inner dim (1)
            represents the sum of squared differences in scores across features
        sums: np.ndarray, float
            2-D array of sums of DeepLIFT contribution scores with shape
            (num_class, num_feature); the outer (0) dim represents the pair of
            compared classes, and the inner dim (1) represents the sum of
            differences in scores across features
        pairs: [(int, int)]
            list of pairs of classes which were compared during interpretation
    """
    # find unique pairs
    pairs = [[(i, j) for j in range(i + 1, num_class)]
             for i in range(num_class)]
    pairs = [p for sublist in pairs for p in sublist]  # flatten
    num_pair = len(pairs)

    # array of running sums of differences (D) and D^2 (D2)
    # for each pair (row) for each feature (column)
    running_sums_D = np.zeros((num_pair, num_feature))
    running_sums_D2 = np.zeros((num_pair, num_feature))
    # array of running sums of contribution scores
    # for each class (row) for each feature (column)
    running_sums_contribs = np.zeros((num_class, num_feature))

    # compute running sums for each pair of classes and their D, D2 values,
    # updating these values batch-by-batch
    for _, batch_contrib_scores in enumerate(generator):
        for class_idx in range(num_class):
            contribs = batch_contrib_scores[class_idx]

            # if only 1 row (e.g., vector), do not sum, will sum all elements
            if contribs.ndim > 1:
                contribs = np.sum(contribs, axis=0)

            running_sums_contribs[class_idx] = np.add(
                running_sums_contribs[class_idx], contribs)

        for pair_idx, (i, j) in enumerate(pairs):
            D = np.subtract(batch_contrib_scores[i], batch_contrib_scores[j])
            D2 = np.square(D)

            # if only 1 row (e.g., vector), do not sum, will sum all elements
            assert D.ndim == D2.ndim
            if D.ndim > 1:
                D = np.sum(D, axis=0)
                D2 = np.sum(D2, axis=0)

            assert D.shape == (num_feature, )
            assert D2.shape == (num_feature, )

            running_sums_D[pair_idx] = np.add(running_sums_D[pair_idx], D)
            running_sums_D2[pair_idx] = np.add(running_sums_D2[pair_idx], D2)

    return running_sums_D, running_sums_D2, running_sums_contribs, pairs


def _paired_ttest_with_diff_sums(sums_D, sums_D2, pairs, num_sample):
    """Performs paired t-tests with sums of differences, D and D^2.

    Arguments:
        sums_D: np.ndarray, float
            2-D array of sums of differences with shape (num_pair, num_feature);
            the outer (0) dim represents the pair of compared classes, and the
            inner dim (1) represents the sum of differences across features
        sums_D2: np.ndarray, float
            2-D array of sums of squared differences with shape
            (num_pair, num_feature); the outer (0) dim represents the pair of
            compared classes, and the inner dim (1) represents the sum of
            squared differences in scores features
        pairs: [(int, int)]
            list of pairs of classes which were compared during interpretation
        num_sample: int
            number of samples

    Returns:
        unadjusted_t_values: np.ndarray, float
            2-D array of unadjusted T values with shape (num_pair, num_feature);
            the outer (0) dim represents the pair of compared classes, and the
            inner dim (1) represents the T value across features
        p_values: np.ndarray, float
            2-D array of adjusted p-values with shape (num_pair, num_feature);
            the outer (0) dim represents the pair of compared classes, and the
            inner dim (1) represents the adjusted p-value across features
    """
    num_pair = len(pairs)
    num_feature = len(sums_D[0])

    # compute T for each pair of classes
    unadjusted_t_values = np.empty((num_pair, num_feature))  # placeholder

    for pair_idx in range(len(pairs)):
        sum_D = sums_D[pair_idx]
        sum_D2 = sums_D2[pair_idx]

        assert np.all(~np.isnan(sum_D))
        assert np.all(~np.isnan(sum_D2))

        N = float(num_sample)
        N_minus_1 = float(num_sample - 1)

        # paired t-test formula from sums of differences
        t = sum_D / np.sqrt((sum_D2 * N - sum_D * sum_D) / N_minus_1)

        unadjusted_t_values[pair_idx] = t

    dof = num_sample - 1  # degrees of freedom

    # compute two-sided p-value, e.g., Pr(abs(t)> tt)
    unadjusted_p_values = stats.t.sf(np.abs(unadjusted_t_values), dof) * 2
    assert unadjusted_p_values.shape == (num_pair, num_feature)

    # apply Bonferroni adjustment to p-values (multiply by # comparisons)
    num_comparison = len(pairs) * num_feature
    p_values = _bonferroni(unadjusted_p_values, num_comparison=num_comparison)
    assert p_values.shape == (num_pair, num_feature)

    return unadjusted_t_values, p_values


def _bonferroni(p_values, num_comparison):
    """Applies Bonferroni adjustment to p-values.

    Arguments:
        p_values: np.ndarray, float
            array of p-values
        num_comparison:
            number of comparisons

    Returns:
        adjusted_p_values: np.ndarray, float
            array of adjusted p-values with the same shape as p_values
    """
    adjust = np.vectorize(lambda pv: min(1.0, pv * num_comparison))
    adjusted_p_values = adjust(p_values)

    assert np.all(adjusted_p_values[~np.isnan(adjusted_p_values)] <= 1.0)
    assert np.all(adjusted_p_values[~np.isnan(adjusted_p_values)] >= 0.0)

    return adjusted_p_values


def _get_list_signif_scores(unadjusted_t_values, p_values):
    """Creates two flattened lists of unadjusted T and adjusted p-values.

    Flattens arrays so that scores corresponding to the same pair of compared
    classes are contiguous, e.g., [f0_p0, f1_p0, f2_p0, f0_p1, f1_p1, ...].

    Arguments:
        unadjusted_t_values: np.ndarray, float
            2-D array of unadjusted T values with shape (num_pair, num_feature);
            the outer (0) dim represents the pair of compared classes, and the
            inner dim (1) represents the T value across features
        p_values: np.ndarray, float
            2-D array of adjusted p-values with shape (num_pair, num_feature);
            the outer (0) dim represents the pair of compared classes, and the
            inner dim (1) represents the adjusted p-value across features

    Returns:
        list_unadjusted_t [float]
            list of unadjusted T values with length num_feature * num_pair
        list_p: [float]
            list of adjusted p-values with length num_feature * num_pair
    """
    num_pair = unadjusted_t_values.shape[0]
    num_feature = unadjusted_t_values.shape[1]

    # flatten nested lists ('C' for row-major, e.g. C style)
    # e.g., np.array([[1, 2, 3], [4, 5, 6]]) => np.array([1, 2, 3, 4, 5, 6])
    # e.g., corresponds to concatenated rows [row0_col0, row1_col0, row2_col0,
    #       row0_col1, row1_col1, row2_col1, row0_col2, row1_col2, row2_col2]
    flat_utv = unadjusted_t_values.flatten('C')
    flat_pv = p_values.flatten('C')

    assert flat_utv.shape == (num_feature * num_pair, )
    assert flat_pv.shape == (num_feature * num_pair, )

    return flat_utv.tolist(), flat_pv.tolist()


def _get_list_pairs(pairs, idx_class_dict, num_feature):
    """Creates flattened list of (repeated) pairs.

    The indexing corresponds with the flattened list of T values and the
    flattened list of p-values obtained from _get_list_signif_scores().

    Arguments:
        pairs: [(int, int)]
            list of pairs of classes which were compared during interpretation
        idx_class_dict: {int: string}
            dictionary mapping class indices to classes
        num_feature: int
            number of features

    Returns:
        list_pairs: [(string, string)]
            list of pairs of compared classes with length num_feature * num_pair
    """
    list_pairs = [[p] * num_feature for p in pairs]
    list_pairs = [p for sublist in list_pairs for p in sublist]  # flatten
    list_pairs = [[idx_class_dict[p[0]], idx_class_dict[p[1]]]
                  for p in list_pairs]  # lookup class
    return list_pairs


def _get_list_feat_names(idx_feat_dict, num_pair):
    """Creates flattened list of (repeated) feature names.

    The indexing corresponds with the flattened list of T values and the
    flattened list of p-values obtained from _get_list_signif_scores().

    Arguments:
        idx_feat_dict: {int: string}
            dictionary mapping feature indices to faetures
        num_class: int
            number of classes

    Returns:
        list_feat_names: [string]
            list of feature names with length num_feature * num_pair
    """
    num_feature = len(idx_feat_dict)
    return [idx_feat_dict[feat_idx] for feat_idx in range(num_feature)] \
        * num_pair


def _get_list_feat_descripts(list_feat_names, icd9_descript_dict):
    """Creates flattened list of (repeated) feature descriptions.

    The indexing corresponds with the flattened list of T values and the
    flattened list of p-values obtained from _get_list_signif_scores().

    Arguments:
        list_feat_names: [string]
            list of feature names corresponding with length
            num_feature * num_pair
        icd9_descript_dict: {string: string}
            dictionary mapping ICD9 codes to description text

    Returns:
        list_feat_descripts: [string]
            list of feature descriptions with length num_feature * num_pair
    """
    # returns the description for a feature; expects the string feature name
    def _get_descript(feat, icd9_descript_dict):
        if feat[:6] == 'gender':
            return 'gender'
        elif feat[:3] == 'age':
            return 'age on record'
        elif feat in icd9_descript_dict:
            return icd9_descript_dict[feat]

        raise ValueError('`{}` not age/gender; not found in icd9_descript_dict'
                         .format(feat))

    list_feat_descripts = [
        _get_descript(f, icd9_descript_dict=icd9_descript_dict)
        for f in list_feat_names]

    return list_feat_descripts
