"""riddle.py

Run various deep learning classification pipelines with k-fold
cross-validation. Summarize discriminatory features using DeepLIFT contribution
scores and paired t-tests with Bonferroni adjustment for multiple comparisons.

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

from utils import evaluate
from utils import get_base_out_dir
from utils import get_param_path
from utils import get_preprocessed_data
from utils import recursive_mkdir
from utils import select_features
from utils import subset_reencode_features


SEED = 109971161161043253 % 8085

parser = argparse.ArgumentParser(
    description='Run RIDDLE (deep classification pipeline).')
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
    '--which_half', type=str, default='both',
    help='Which half of experiments to perform; values = first, last, both')
parser.add_argument(
    '--data_dir', type=str, default='_data',
    help='Directory of data files.')
parser.add_argument(
    '--cache_dir', type=str, default='_cache',
    help='Directory where to cache files.')
parser.add_argument(
    '--out_dir', type=str, default='_out',
    help='Directory where to save output files.')


def run(x_unvec, y, idx_feat_dict, idx_class_dict, icd9_descript_dict,
        num_feature, max_num_feature, num_class, feature_selection, k_idx, k,
        params, perm_indices, full_out_dir):
    """Run a RIDDLE classification pipeline for a single k-fold partition.

    Arguments:
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
        num_feature: int
            number of features present in the dataset
        max_num_feature: int
            maximum number of features to use
        num_class: int
            number of classes
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
        full_out_dir: string
            directory where outputs (e.g., results) should be saved
    """
    from keras import backend as K
    from riddle import emr, feature_importance
    from riddle.models import MLP

    print('Partition k = {}'.format(k_idx))
    print()
    x_train_unvec, y_train, x_val_unvec, y_val, x_test_unvec, y_test = (
        emr.get_k_fold_partition(x_unvec, y, k_idx=k_idx, k=k,
                                 perm_indices=perm_indices))

    if max_num_feature > 0:  # select features and re-encode
        feat_encoding_dict, idx_feat_dict = select_features(
            x_train_unvec, y_train, idx_feat_dict,
            method=feature_selection, num_feature=num_feature,
            max_num_feature=max_num_feature)
        x_train_unvec = subset_reencode_features(x_train_unvec,
                                                 feat_encoding_dict)
        x_val_unvec = subset_reencode_features(x_val_unvec,
                                               feat_encoding_dict)
        x_test_unvec = subset_reencode_features(x_test_unvec,
                                                feat_encoding_dict)
        num_feature = max_num_feature

    # set up
    max_num_epoch = -1
    if 'debug' in full_out_dir:
        max_num_epoch = 3
    model = MLP(num_feature=num_feature, num_class=num_class,
                max_num_epoch=max_num_epoch, **params[k_idx])

    # train and test
    start = time.time()

    model.train(x_train_unvec, y_train, x_val_unvec, y_val)
    y_test_probas = model.predict_proba(x_test_unvec)

    runtime = time.time() - start
    print('Completed training and testing in {:.4f} seconds'.format(runtime))
    print('-' * 72)
    print()

    # evaluate model performance
    evaluate(y_test, y_test_probas, runtime, num_class=num_class,
             out_dir=full_out_dir)

    model.save_model(path=full_out_dir + '/model.h5')
    K.clear_session()

    print('Finished with partition k = {}'.format(k_idx))
    print('=' * 72)
    print()


def run_kfold(data_fn, prop_missing=0., max_num_feature=-1,
              feature_selection='random', k=10, which_half='both',
              data_dir='_data', cache_dir='_cache', out_dir='_out'):
    """Run several RIDDLE classification pipelines a la k-fold cross-validation.

    Arguments:
        data_fn: string
            data file filename
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
            outer directory where outputs (e.g., results) should be saved
    """
    start = time.time()

    base_out_dir = get_base_out_dir(out_dir, 'riddle', data_fn, prop_missing,
                                    max_num_feature, feature_selection)
    recursive_mkdir(base_out_dir)

    # get common data
    x_unvec, y, idx_feat_dict, idx_class_dict, icd9_descript_dict, perm_indices = (
        get_preprocessed_data(data_dir, data_fn, prop_missing=prop_missing))
    num_feature = len(idx_feat_dict)
    num_class = len(idx_class_dict)

    # print/save value-sorted dictionary of classes and features
    class_mapping = sorted(idx_class_dict.items(), key=lambda key: key[0])
    with open(base_out_dir + '/class_mapping.txt', 'w') as f:
        print(class_mapping, file=f)
    with open(base_out_dir + '/feature_mapping.txt', 'w') as f:
        for idx, feat in idx_feat_dict.items():
            f.write('{}\t{}\n'.format(idx, feat))

    try:  # load saved parameters
        param_path = get_param_path(cache_dir, 'riddle', data_fn, prop_missing,
                                    max_num_feature, feature_selection)
        with open(param_path, 'rb') as f:
            params = pickle.load(f)

        # for legacy compatability
        new_params = {}
        for k_idx, param in params.items():
            if 'nb_hidden_layers' in param:
                param['num_hidden_layer'] = param.pop('nb_hidden_layers')
            if 'nb_hidden_nodes' in param:
                param['num_hidden_node'] = param.pop('nb_hidden_nodes')
            new_params[k_idx] = param
        params = params

    except:
        warnings.warn('Cannot load parameters from: {}\n'.format(param_path) +
                      'Need to do parameter search; run parameter_search.py')
        raise

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

        run(x_unvec, y, idx_feat_dict, idx_class_dict, icd9_descript_dict,
            num_feature=num_feature, max_num_feature=max_num_feature,
            num_class=num_class, feature_selection=feature_selection,
            k_idx=k_idx, k=k, params=params, perm_indices=perm_indices,
            full_out_dir=sub_out_dir)

    print('This k-fold riddle multipipeline run script took {:.4f} seconds'
          .format(time.time() - start))


def main():
    """Main method."""
    np.random.seed(SEED)  # for reproducibility, must be before Keras imports!
    run_kfold(data_fn=FLAGS.data_fn,
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
