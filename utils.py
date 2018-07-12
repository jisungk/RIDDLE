"""utils.py

Offers various useful utility functions for experiment scripts.

Requires:   NumPy, scikit-learn, SciPy, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

from collections import Counter
import os
import pickle
import time
import warnings

import numpy as np
import scipy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss

from riddle import emr

PRINT_PRECISION = 4


def get_data_path(data_dir, data_fn):
    """Get path of data file."""
    return '{}/{}'.format(data_dir, data_fn)


def get_icd9_descript_path(data_dir):
    """Get path of icd9 description file."""
    return '{}/{}'.format(data_dir, 'phewas_codes.txt')


def get_param_path(cache_dir, method, data_fn, prop_missing, max_num_feature,
                   feature_selection):
    """Get path of pickled parameters."""
    return (
        '{}/{}_datafn={}_propmiss={}_maxfeat={}_featselect={}_param.pkl'.format(
            cache_dir, method, data_fn, prop_missing, max_num_feature,
            feature_selection))


def get_base_out_dir(out_dir, method, data_fn, prop_missing, max_num_feature,
                     feature_selection):
    """Get path of pickled parameters."""
    base_out_dir = (
        '{}/{}_datafn={}_propmiss={}_maxfeat={}_featselect={}'.format(
            out_dir, method, data_fn, prop_missing, max_num_feature,
            feature_selection))
    return base_out_dir


def get_perm_indices_path(data_dir, data_fn):
    """Get path of pickled perm_indices file."""
    return '{}/{}_perm_indices.pkl'.format(data_dir, data_fn)


def recursive_mkdir(dir_path):
    """Create directory folders in a recursive fashion."""
    curr_path = ''
    for subfolder in dir_path.split('/'):
        curr_path += subfolder
        if curr_path != '' and not os.path.exists(curr_path):
            os.makedirs(curr_path)
        curr_path += '/'

    assert curr_path.rstrip('/') == dir_path.rstrip('/').lstrip('/')


def evaluate(y_test, y_test_probas, runtime, num_class, out_dir):
    """Perform and save a detailed evaluation of prediction results.

    Computes a confusion matrix, various scores and ROC curves.

    Arguments:
        y_test: [int]
            list of test class labels with length num_sample
        y_test_probas: np.ndarray, float
            array of predicted probabilities with shape (num_sample, num_class)
        runtime: float
            running time
        num_class: int
            number of classes
        out_dir:
            directory where outputs (e.g., results) should be saved
    """
    from riddle import roc  # np must be seeded before importing riddle & Keras

    # save predictions
    predictions_path = out_dir + '/test_predictions.txt'
    with open(predictions_path, 'w') as f:
        for prob_vector, target in zip(y_test_probas, y_test):
            for idx, p in enumerate(prob_vector):
                if idx != 0:
                    f.write('\t')
                f.write(str(p))
            f.write('\t')
            f.write(str(target))
            f.write('\n')

    y_pred = np.argmax(y_test_probas, axis=1)
    report = classification_report(y_test, y_pred, digits=5)
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_test_probas)

    roc_path = out_dir + '/roc.png'
    roc_auc_dict, _, _ = roc.compute_and_plot_roc_scores(
        y_test, y_test_probas, num_class=num_class, path=roc_path)

    # save evaluation results
    metrics_path = out_dir + '/test_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write('Classification report\n')
        f.write(report)
        f.write('\n')
        f.write('acc = {:.5f}\n'.format(acc))
        f.write('loss = {:.5f}\n'.format(loss))
        f.write('runtime = {:.5f}\n'.format(runtime))
        f.write('\n')
        f.write('AUC metrics\n')

        for k, v in roc_auc_dict.items():
            f.write('{} = {:.5f}\n'.format(k, v))


def select_features(x_unvec, y, idx_feat_dict, method, num_feature,
                    max_num_feature):
    """Performs feature selection to reduce the number of features.

    Arguments:
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        y: [int]
            list of class labels as integer indices
        idx_feat_dict: {int: string}
            dictionary mapping feature indices to features
        method: string
            feature selection method
        num_feature: int
            number of features present in dataset
        max_num_features: int
            maximum number of features to use

    Returns:
        feat_encoding_dict: {int: int}
            dictionary mapping old feature indices to new feature indices
        new_idx_feat_dict:
            dictionary mapping new feature indices to features
    """
    if max_num_feature < 1:
        raise ValueError('Tried to select a number of features which is < 1.')

    num_feature = len(idx_feat_dict)

    if max_num_feature >= num_feature:  # no feature selection
        warnings.warn('Skipping feature selection since the number of desired'
                      'features is greater than the number of actual features')
        return {i: i for i in range(num_feature)}, idx_feat_dict
    if method == 'random':
        feature_idxs = idx_feat_dict.keys()
        np.random.shuffle(feature_idxs)
        selected_features = feature_idxs[:max_num_feature]
    elif method == 'frequency':
        # does not assume features which have lower indices are more frequent
        feature_counts = Counter([i for row in x_unvec for i in row])
        selected_features = [feature for feature, _ in
                             feature_counts.most_common(max_num_feature)]
    elif method == 'chi2':
        x = vectorize_features(x_unvec, num_feature)
        feature_selector = SelectKBest(chi2, k=max_num_feature).fit(x, y)
        selected_features = feature_selector.get_support(indices=True)
    else:
        raise ValueError('Unknown feature selection method: {}'.format(method))

    print('Reduced {} features to {} using: {}'.format(
        num_feature, len(selected_features), method))

    # get dictionaries to re-encode features with new indices
    feat_encoding_dict = {old_feat_idx: new_feat_idx for
                          new_feat_idx, old_feat_idx in
                          enumerate(sorted(selected_features))}
    new_idx_feat_dict = {new_feat_idx: idx_feat_dict[old_feat_idx] for
                         old_feat_idx, new_feat_idx in
                         feat_encoding_dict.items()}

    return feat_encoding_dict, new_idx_feat_dict


def subset_reencode_features(x_unvec, feat_encoding_dict):
    """Filter then re-encode features after feature selection.

    Arguments:
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        feat_encoding_dict: {int: int}
            dictionary mapping old feature indices to new feature indices

    Returns:
        x_unvec_new: [[int]]
            modified feature indices that have not been vectorized; each inner
            list collects the indices of features that are present (binary on)
            for a sample
    """
    return [[feat_encoding_dict[feat_idx] for feat_idx in x if
             feat_idx in feat_encoding_dict] for x in x_unvec]


def vectorize_features(x_unvec, num_feature):
    """Vectorizes feature data via binary encodin to a sparse scipy matrix.

    This is applied to conform to the scikit-learn API.

    Arguments:
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        num_feature: int
            number of features

    Returns:
        x_vec: scipy.sparse.*matrix
            sparse scipy matrix of binary features with shape
            (num_sample, num_feature)
    """
    num_sample = len(x_unvec)
    x_vec = scipy.sparse.lil_matrix((num_sample, num_feature))

    for row_idx, x in enumerate(x_unvec):
        x_vec[row_idx, sorted(x)] = 1

    return x_vec


def get_preprocessed_data(data_dir, data_fn, prop_missing=0.0):
    """Get preprocessed data for a machine learning pipeline.

    Arguments:
        data_dir: string
            directory where data files are located
        data_fn: string
            filename of data file
        prop_missing: float
            proportion of feature observations which should be randomly masked;
            values in [0, 1)

    Returns:
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        y: [int]
            list of class labels as integer indices
        idx_feat_dict: {int: string}
            dictionary mapping feature indices to features
        idx_class_dict: {int: string}
            dictionary mapping class indices to classes
        icd9_descript_dict: {string: string}
            dictionary mapping ICD9 codes to description text
        perm_indices: np.ndarray, int
            array of indices representing a permutation of the samples with
            shape (num_sample, )
    """
    print('Loading data...')
    start = time.time()

    icd9_descript_path = get_icd9_descript_path(data_dir)
    data_path = get_data_path(data_dir, data_fn)
    perm_indices_path = get_perm_indices_path(data_dir, data_fn)

    # get common data
    icd9_descript_dict = emr.get_icd9_descript_dict(icd9_descript_path)
    x_unvec, y, idx_feat_dict, idx_class_dict = emr.get_data(
        path=data_path, icd9_descript_dict=icd9_descript_dict)
    num_sample = len(x_unvec)

    # must be before any other calls to np.random
    perm_indices = np.random.permutation(num_sample)
    if os.path.isfile(perm_indices_path):
        with open(perm_indices_path, 'rb') as f:
            expected_perm_indices = pickle.load(f)
        assert np.all(perm_indices == expected_perm_indices)
    else:  # cache perm_indices
        with open(perm_indices_path, 'wb') as f:
            pickle.dump(perm_indices, f)

    if prop_missing != 0.0:
        x_unvec = _simulate_missing_data(x_unvec, prop_missing=prop_missing)

    print('Data loaded in {:.5f} s'.format(time.time() - start))
    print()

    return (x_unvec, y, idx_feat_dict, idx_class_dict, icd9_descript_dict,
            perm_indices)


def _simulate_missing_data(x_unvec, prop_missing):
    """Removes a proportion of feature observations completely at random.

    Does not remove whole features but rather individual feature occurences.

    Arguments:
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        prop_missing: float
            proportion of feature observations which should be randomly masked;
            values in [0, 1)

    Returns:
        x_unvec: [[int]]
            modified feature indices that have not been vectorized; each inner
            list collects the indices of features that are present (binary on)
            for a sample
    """
    if prop_missing < 0:
        raise ValueError(
            'prop_missing for simulating missing data is negative')
    if prop_missing == 0.0:
        return x_unvec

    # sparse_col_idx is not the encoded index of the feature but rather the
    # literal position of that feature observation within a sample
    indices = [[(row_idx, sparse_col_idx) for sparse_col_idx in range(len(x))]
               for row_idx, x in enumerate(x_unvec)]
    indices = [i for r in indices for i in r]  # flatten
    np.random.shuffle(indices)

    num_to_remove = int(len(indices) * prop_missing)
    indices_to_remove = indices[:int(len(indices) * prop_missing)]
    # need to sort these so that deletion works and doesn't screw up indices
    indices_to_remove = sorted(indices_to_remove, key=lambda x: (x[0], x[1]),
                               reverse=True)

    for row_idx, sparse_col_idx in indices_to_remove:
        x_unvec[row_idx].pop(sparse_col_idx)

    # check number of removals
    lengths = [len(x) for x in x_unvec]
    assert sum(lengths) == len(indices) - num_to_remove

    print('Randomly deleted {} feature occurrences'.format(num_to_remove))

    return x_unvec
