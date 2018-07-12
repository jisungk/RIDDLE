"""other_clf.py

Run various machine learning classification pipelines with k-fold
cross-validation.

Requires:   Keras, NumPy, scikit-learn, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

from __future__ import print_function

import argparse
import pickle
import time
import warnings

import numpy as np

from riddle import emr

from utils import evaluate
from utils import get_base_out_dir
from utils import get_param_path
from utils import get_preprocessed_data
from utils import recursive_mkdir
from utils import select_features
from utils import subset_reencode_features
from utils import vectorize_features

SEED = 109971161161043253 % 8085

parser = argparse.ArgumentParser(
    description='Perform parameter search for various classification methods.')
parser.add_argument(
    '--method', type=str, default='logit',
    help='Classification method to use.')
parser.add_argument(
    '--data_fn', type=str, default='debug.txt',
    help='Filename of text data file.')
parser.add_argument(
    '--prop_missing', type=float, default=0.0,
    help='Proportion of feature observations to simulate as missing.')
parser.add_argument(
    '--max_num_feature', type=int, default=-1,
    help='Maximum number of features to use; with the default of -1, use all'
         'available features')
parser.add_argument(
    '--feature_selection', type=str, default='random',
    help='Method to use for feature selection.')
parser.add_argument(
    '--which_half', type=str, default='both',
    help='Which half of experiments to perform; values = first, last, both')
parser.add_argument(
    '--data_dir', type=str, default='_data',
    help='Directory of data files.')
parser.add_argument(
    '--cache_dir', type=str, default='_cache',
    help='Directory where to cache files and outputs.')
parser.add_argument(
    '--out_dir', type=str, default='_out',
    help='Directory where to save output files.')


def run(ModelClass, x_unvec, y, idx_feat_dict, num_feature, max_num_feature,
        num_class, feature_selection, k_idx, k, params, perm_indices,
        init_args, full_out_dir):
    """Run a classification pipeline for a single k-fold partition.

    Arguments:
        ModelClass: Python class
            classification model
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        y: [int]
            list of class labels as integer indices
        idx_feat_dict: {int: string}
            dictionary mapping feature indices to features
        num_feature: int
            number of features present in the dataset
        max_num_feature: int
            maximum number of features to use
        num_class: int
            number of classes present
        feature_selection: string
            feature selection method; values = {'random', 'frequency', 'chi2'}
        k_idx: int
            index of the k-fold partition to use
        k: int
            number of partitions for k-fold cross-validation
        params: [{string: ?}]
            list of dictionary mapping parameter names to values for each
            k-fold partition
        perm_indices: np.ndarray, int
            array of indices representing a permutation of the samples with
            shape (num_sample, )
        init_args: {string: ?}
            dictionary mapping initialization argument names to values
        out_dir: string
            directory where outputs (e.g., results) should be saved
    """
    print('-' * 72)
    print('Partition k = {}'.format(k_idx))
    print(params[k_idx])

    x_train_unvec, y_train, _, _, x_test_unvec, y_test = (
        emr.get_k_fold_partition(x_unvec, y, k_idx=k_idx, k=k,
                                 perm_indices=perm_indices))

    if max_num_feature > 0:  # select features and re-encode
        feat_encoding_dict, _ = select_features(
            x_train_unvec, y_train, idx_feat_dict,
            method=feature_selection, num_feature=num_feature,
            max_num_feature=max_num_feature)
        x_train_unvec = subset_reencode_features(
            x_train_unvec, feat_encoding_dict)
        x_test_unvec = subset_reencode_features(
            x_test_unvec, feat_encoding_dict)
        num_feature = max_num_feature

    x_train = vectorize_features(x_train_unvec, num_feature)
    x_test = vectorize_features(x_test_unvec, num_feature)

    args = dict(init_args)  # copy dictionary
    args.update(params[k_idx])

    start = time.time()
    model = ModelClass(**args)
    model.fit(x_train, y_train)
    y_test_probas = model.predict_proba(x_test)
    runtime = time.time() - start

    evaluate(y_test, y_test_probas, runtime, num_class=num_class,
             out_dir=full_out_dir)


def run_kfold(data_fn, method='logit', prop_missing=0., max_num_feature=-1,
              feature_selection='random', k=10, which_half='both',
              data_dir='_data', cache_dir='_cache', out_dir='_out'):
    """Run several classification pipelines a la k-fold cross-validation.

    Arguments:
        data_fn: string
            data file filename
        method: string
            name of classification method; values = {'logit', 'random_forest',
            'linear_svm', 'poly_svm', 'rbf_svm', 'gbdt'}
        prop_missing: float
            proportion of feature observations which should be randomly masked;
            values in [0, 1)
        max_num_feature: int
            maximum number of features to use
        feature_selection: string
            feature selection method; values = {'random', 'frequency', 'chi2'}
        k: int
            number of partitions for k-fold cross-validation
        which_half: str
            which half of experiments to do; values = {'first', 'last', 'both'}
        data_dir: string
            directory where data files are located
        cache_dir: string
            directory where cached files (e.g., saved parameters) are located
        out_dir: string
            directory where
        perm_indices: np.ndarray, int
            array of indices representing a permutation of the samples with
            shape (num_sample, )
        init_args: {string: ?}
            dictionary mapping initialization argument names to values
        out_dir: string
            directory where outputs (e.g., results) should be saved
    """
    start = time.time()

    try:  # load saved parameters
        param_path = get_param_path(cache_dir, method, data_fn, prop_missing,
                                    max_num_feature, feature_selection)
        with open(param_path, 'rb') as f:
            params = pickle.load(f)
    except:
        warnings.warn('Cannot load parameters from: {}\n'.format(param_path) +
                      'Need to do parameter search; run parameter_search.py')
        raise

    # TODO(jisungkim) handle binary and multiclass separately, don't assume
    # multiclass!
    if method == 'logit':
        from sklearn.linear_model import LogisticRegression as ModelClass
        init_args = {'multi_class': 'multinomial', 'solver': 'lbfgs'}
    elif method == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier as ModelClass
        init_args = {}
    elif method == 'linear_svm':
        from sklearn.svm import SVC as ModelClass
        # remark: due to a bug in scikit-learn / libsvm, the sparse 'linear'
        # kernel is much slower than the sparse 'poly' kernel, so we use
        # the 'poly' kernel with degree=1 over the 'linear' kernel
        init_args = {'kernel': 'poly', 'degree': 1, 'coef0': 0.,
                     'gamma': 1., 'probability': True, 'cache_size': 1000}
    elif method == 'poly_svm':
        from sklearn.svm import SVC as ModelClass
        init_args = {'kernel': 'poly', 'probability': True, 'cache_size': 1000}
    elif method == 'rbf_svm':
        from sklearn.svm import SVC as ModelClass
        init_args = {'kernel': 'rbf', 'probability': True, 'cache_size': 1000}
    elif method == 'gbdt':
        from xgboost import XGBClassifier as ModelClass
        init_args = {'objective': 'multi:softprob'}
    else:
        raise ValueError('unknown method: {}'.format(method))

    x_unvec, y, idx_feat_dict, idx_class_dict, _, perm_indices = (
        get_preprocessed_data(data_dir, data_fn, prop_missing=prop_missing))
    num_feature = len(idx_feat_dict)
    num_class = len(idx_class_dict)

    base_out_dir = get_base_out_dir(out_dir, method, data_fn, prop_missing,
                                    max_num_feature, feature_selection)
    recursive_mkdir(base_out_dir)

    if which_half == 'both':
        loop = range(0, k)
    elif which_half == 'first':
        loop = range(0, k / 2)
    elif which_half == 'last':
        loop = range(k / 2, k)
    else:
        raise ValueError('Unknown which_half: {}'.format(which_half))

    for k_idx in loop:
        sub_out_dir = '{}/k_idx={}'.format(base_out_dir, k_idx)
        recursive_mkdir(sub_out_dir)

        run(ModelClass, x_unvec, y, idx_feat_dict, num_feature=num_feature,
            max_num_feature=max_num_feature, num_class=num_class,
            feature_selection=feature_selection, k_idx=k_idx, k=k,
            params=params, perm_indices=perm_indices, init_args=init_args,
            full_out_dir=sub_out_dir)

    print('This k-fold {} multipipeline run script took {:.4f} seconds'
          .format(method, time.time() - start))


def main():
    """Main method."""
    np.random.seed(SEED)  # for reproducibility, must be before Keras imports!
    run_kfold(data_fn=FLAGS.data_fn,
              method=FLAGS.method,
              prop_missing=FLAGS.prop_missing,
              max_num_feature=FLAGS.max_num_feature,
              feature_selection=FLAGS.feature_selection,
              which_half=FLAGS.which_half,
              data_dir=FLAGS.data_dir,
              cache_dir=FLAGS.cache_dir,
              out_dir=FLAGS.out_dir)


# if run as script, execute main
if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    main()
