"""parameter_search.py

Search for optimal parameters for RIDDLE and various ML classifiers.

Requires:   Keras, NumPy, scikit-learn, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

from __future__ import print_function

import argparse
import os
import pickle
import time
import warnings

import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import RandomizedSearchCV

from riddle import emr
from riddle import tuning
from riddle.models import MLP

from utils import get_param_path
from utils import get_preprocessed_data
from utils import recursive_mkdir
from utils import select_features
from utils import subset_reencode_features
from utils import vectorize_features

SEED = 109971161161043253 % 8085
TUNING_K = 3  # number of partitions to use to evaluate a parameter config

parser = argparse.ArgumentParser(
    description='Perform parameter search for various classification methods.')
parser.add_argument(
    '--method', type=str, default='riddle',
    help='Classification method to use.')
parser.add_argument(
    '--data_fn', type=str, default='dummy.txt',
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
    '--force_run', type=bool, default=False,
    help='Whether to force parameter search to run even if it has been already'
         'performed.')
parser.add_argument(
    '--max_num_sample', type=int, default=10000,
    help='Maximum number of samples to use during parameter tuning.')
parser.add_argument(
    '--num_search', type=int, default=5,
    help='Number of parameter settings (searches) to try.')
parser.add_argument(
    '--data_dir', type=str, default='_data',
    help='Directory of data files.')
parser.add_argument(
    '--cache_dir', type=str, default='_cache',
    help='Directory where to cache files and outputs.')


def loss_scorer(estimator, x, y):
    """Negative log loss scoring function for scikit-learn model selection."""
    loss = log_loss(y, estimator.predict_proba(x))
    assert loss >= 0
    # we want to minimize loss; since scikit-learn model selection tries to
    # maximize a given score, return the negative of the loss
    return -1 * loss


def run(method, x_unvec, y, idx_feat_dict, num_feature, max_num_feature,
        num_class, max_num_sample, feature_selection, k_idx, k, num_search,
        perm_indices):
    """Run a parameter search for a single k-fold partitions

    Arguments:
        method: string
            name of classification method; values = {'logit', 'random_forest',
            'linear_svm', 'poly_svm', 'rbf_svm', 'gbdt', 'riddle'}
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
        num_search: int
            number of searches (parameter configurations) to try
        perm_indices: np.ndarray, int
            array of indices representing a permutation of the samples with
            shape (num_sample, )

    Returns:
        best_param: {string: ?}
            dictionary mapping parameter names to the best values found
    """
    print('-' * 72)
    print('Partition k = {}'.format(k_idx))

    x_train_unvec, y_train, x_val_unvec, y_val, _, _ = (
        emr.get_k_fold_partition(x_unvec, y, k_idx=k_idx, k=k,
                                 perm_indices=perm_indices))

    if max_num_feature > 0:  # select features and re-encode
        feat_encoding_dict, _ = select_features(
            x_train_unvec, y_train, idx_feat_dict,
            method=feature_selection, num_feature=num_feature,
            max_num_feature=max_num_feature)
        x_val_unvec = subset_reencode_features(x_val_unvec, feat_encoding_dict)
        num_feature = max_num_feature

    # cap number of validation samples
    if max_num_sample != None and len(x_val_unvec) > max_num_sample:
        x_val_unvec = x_val_unvec[0:max_num_sample]
        y_val = y_val[0:max_num_sample]

    start = time.time()
    if method == 'riddle':
        model_class = MLP
        init_args = {'num_feature':num_feature, 'num_class': num_class}
        param_dist = {
            'num_hidden_layer': 2,  # [1, 2]
            'num_hidden_node': 512,  # [128, 256, 512]
            'activation': ['prelu', 'relu'],
            'dropout': tuning.Uniform(lo=0.2, hi=0.8),
            'learning_rate': tuning.UniformLogSpace(10, lo=-6, hi=-1),
            }
        best_param = tuning.random_search(
            model_class, init_args, param_dist, x_val_unvec, y_val,
            num_class=num_class, k=TUNING_K, num_search=num_search)
    else:  # scikit-learn methods
        x_val = vectorize_features(x_val_unvec, num_feature)

        if method == 'logit':  # logistic regression
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(multi_class='multinomial',
                                           solver='lbfgs')
            param_dist = {'C': tuning.UniformLogSpace(base=10, lo=-3, hi=3)}
        elif method == 'random_forest':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier()
            param_dist = {
                'max_features': ['sqrt', 'log2', None],
                'max_depth': tuning.UniformIntegerLogSpace(base=2, lo=0, hi=7),
                'n_estimators': tuning.UniformIntegerLogSpace(base=2, lo=4, hi=8)
                }
        elif method == 'linear_svm':
            from sklearn.svm import SVC
            # remark: due to a bug in scikit-learn / libsvm, the sparse 'linear'
            # kernel is much slower than the sparse 'poly' kernel, so we use
            # the 'poly' kernel with degree=1 over the 'linear' kernel
            estimator = SVC(kernel='poly', degree=1, coef0=0., gamma=1.,
                            probability=True, cache_size=1000)
            param_dist = {
                'C': tuning.UniformLogSpace(base=10, lo=-2, hi=1)
                }
        elif method == 'poly_svm':
            from sklearn.svm import SVC
            estimator = SVC(kernel='poly', probability=True, cache_size=1000)
            param_dist = {
                'C': tuning.UniformLogSpace(base=10, lo=-2, hi=1),
                'degree': [2, 3, 4],
                'gamma': tuning.UniformLogSpace(base=10, lo=-5, hi=1)
                }
        elif method == 'rbf_svm':
            from sklearn.svm import SVC
            estimator = SVC(kernel='rbf', probability=True, cache_size=1000)
            param_dist = {
                'C': tuning.UniformLogSpace(base=10, lo=-2, hi=1),
                'gamma': tuning.UniformLogSpace(base=10, lo=-5, hi=1)
                }
        elif method == 'gbdt':
            from xgboost import XGBClassifier
            estimator = XGBClassifier(objective='multi:softprob')
            param_dist = {
                'max_depth': tuning.UniformIntegerLogSpace(base=2, lo=0, hi=5),
                'n_estimators': tuning.UniformIntegerLogSpace(base=2, lo=4, hi=8),
                'learning_rate': tuning.UniformLogSpace(base=10, lo=-3, hi=0)
                }
        else:
            raise ValueError('unknown method: {}'.format(method))

        param_search = RandomizedSearchCV(
            estimator, param_dist, refit=False, n_iter=num_search,
            scoring=loss_scorer)
        param_search.fit(x_val, y_val)


        best_param = param_search.best_params_

    print('Best parameters for {} for k_idx={}: {} found in {:.3f} s'
          .format(method, k_idx, best_param, time.time() - start))

    return best_param


def run_kfold(data_fn, method='logit', prop_missing=0., max_num_feature=-1,
              feature_selection='random', k=10, max_num_sample=10000,
              num_search=30, data_dir='_data', cache_dir='_cache',
              force_run=False):
    """Run several parameter searches a la k-fold cross-validation.

    Arguments:
        data_fn: string
            data file filename
        method: string
            name of classification method; values = {'logit', 'random_forest',
            'linear_svm', 'poly_svm', 'rbf_svm', 'gbdt', 'riddle'}
        prop_missing: float
            proportion of feature observations which should be randomly masked;
            values in [0, 1)
        max_num_feature: int
            maximum number of features to use
        feature_selection: string
            feature selection method; values = {'random', 'frequency', 'chi2'}
        k: int
            number of partitions for k-fold cross-validation
        max_num_sample: int
            maximum number of samples to use
        num_search: int
            number of searches (parameter configurations) to try for each
            partition
        data_dir: string
            directory where data files are located
        cache_dir: string
            directory where cached files (e.g., saved parameters) are located
        out_dir: string
            directory where outputs (e.g., results) should be saved
    """
    if 'debug' in data_fn:
        num_search = 3

    # check if already did param search, if so, skip
    param_path = get_param_path(cache_dir, method, data_fn, prop_missing,
                                max_num_feature, feature_selection)
    if not force_run and os.path.isfile(param_path):
        warnings.warn('Already did search for {}, not performing search'
                      .format(method))
        return

    x_unvec, y, idx_feat_dict, idx_class_dict, _, perm_indices = (
        get_preprocessed_data(data_dir, data_fn, prop_missing=prop_missing))
    num_feature = len(idx_feat_dict)
    num_class = len(idx_class_dict)
    params = {}
    for k_idx in range(0, k):
        params[k_idx] = run(
            method, x_unvec, y, idx_feat_dict, num_feature=num_feature,
            max_num_feature=max_num_feature, num_class=num_class,
            max_num_sample=max_num_sample, feature_selection=feature_selection,
            k_idx=k_idx, k=k, num_search=num_search, perm_indices=perm_indices)

    recursive_mkdir(FLAGS.cache_dir)
    with open(param_path, 'w') as f:  # save
        pickle.dump(params, f)

    print('Finished parameter search for method: {}'.format(method))


def main():
    """Main method."""
    np.random.seed(SEED)  # for reproducibility, must be before Keras imports!
    run_kfold(data_fn=FLAGS.data_fn,
              method=FLAGS.method,
              prop_missing=FLAGS.prop_missing,
              max_num_feature=FLAGS.max_num_feature,
              feature_selection=FLAGS.feature_selection,
              max_num_sample=FLAGS.max_num_sample,
              num_search=FLAGS.num_search,
              data_dir=FLAGS.data_dir,
              cache_dir=FLAGS.cache_dir,
              force_run=FLAGS.force_run)


# if run as script, execute main
if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    main()
