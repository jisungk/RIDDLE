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
from functools import partial
import pickle
import time
import warnings

import numpy as np

from utils import evaluate
from utils import get_base_out_dir
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
    '--interpret_model', type=bool, default=False,
    help='Whether to run model interpretation.')
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


def run_interpretation_summary(x_unvec, y, contrib_sums_D, contrib_sums_D2,
                               contrib_sums, idx_feat_dict, idx_class_dict,
                               icd9_descript_dict, pairs, num_sample,
                               full_out_dir):
    """Summarize DeepLIFT contribution scores to interpret a trained NN.

    Arguments:
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        y: [int]
            list of class labels as integer indices
        contrib_sums_D: np.ndarray, float
            2-D array of sums of differences in DeepLIFT contribution scores
            with shape (num_pair, num_feature); the outer (0) dim represents the
            pair of compared classes, and the inner dim (1) represents the sum
            of differences in scores across features
        contrib_sums_D2: np.ndarray, float
            2-D array of sums of squared differences in DeepLIFT contribution
            scores with shape (num_pair, num_feature); the outer (0) dim
            represents the pair of compared classes, and the inner dim (1)
            represents the sum of squared differences in scores across features
        contrib_sums: np.ndarray, float
            2-D array of sums of DeepLIFT contribution scores with shape
            (num_class, num_feature); the outer (0) dim represents the class and
            the inner dim (1) represents the sum of scores across features
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
        full_out_dir: string
            directory where outputs (e.g., results) should be saved
    """
    from riddle import feature_importance, frequency, ordering

    # get descriptions of feature importance
    # TODO(jisungkim) should actually use this summary
    feat_importance_summary = feature_importance.FeatureImportanceSummary(
        contrib_sums_D, contrib_sums_D2, idx_feat_dict=idx_feat_dict,
        idx_class_dict=idx_class_dict, icd9_descript_dict=icd9_descript_dict,
        pairs=pairs, num_sample=num_sample)

    # get frequencies of features per class
    feat_class_freq_table = frequency.get_frequency_table(
        x_unvec, y, idx_feat_dict=idx_feat_dict, idx_class_dict=idx_class_dict)

    # get orderings
    ordering_summary = ordering.summarize_orderings(
        contrib_sums, feat_class_freq_table, idx_feat_dict=idx_feat_dict,
        idx_class_dict=idx_class_dict, icd9_descript_dict=icd9_descript_dict)

    ordering_summary.save_individual_tables(idx_class_dict, full_out_dir)
    ordering_summary.save(full_out_dir)


def run(data_fn, prop_missing=0., max_num_feature=-1,
        feature_selection='random', k=10, data_dir='_data', out_dir='_out'):
    """Run RIDDLE classification interpretation pipeline.

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
        interpret_model: bool
            whether to interpret the trained model for first k-fold partition
        which_half: str
            which half of experiments to do; values = {'first', 'last', 'both'}
        data_dir: string
            directory where data files are located
        cache_dir: string
            directory where cached files (e.g., saved parameters) are located
        out_dir: string
            outer directory where outputs (e.g., results) should be saved
    """
    from keras.models import load_model
    from riddle import emr, feature_importance
    from riddle.models import MLP

    start = time.time()

    base_out_dir = get_base_out_dir(out_dir, 'riddle', data_fn, prop_missing,
                                    max_num_feature, feature_selection)
    recursive_mkdir(base_out_dir)

    # get common data
    x_unvec, y, idx_feat_dict, idx_class_dict, icd9_descript_dict, perm_indices = (
        get_preprocessed_data(data_dir, data_fn, prop_missing=prop_missing))
    num_feature = len(idx_feat_dict)
    num_class = len(idx_class_dict)

    list_sums_D, list_sums_D2, list_sums_contribs = [], [], []

    for k_idx in range(k):
        full_out_dir = '{}/k_idx={}'.format(base_out_dir, k_idx)
        print('\nPartition k = {}'.format(k_idx))
        x_train_unvec, y_train, _, _, x_test_unvec, y_test = emr.get_k_fold_partition(
            x_unvec, y, k_idx=k_idx, k=k, perm_indices=perm_indices)

        if max_num_feature > 0:  # select features and re-encode
            feat_encoding_dict, idx_feat_dict = select_features(
                x_train_unvec, y_train, idx_feat_dict,
                method=feature_selection, num_feature=num_feature,
                max_num_feature=max_num_feature)
            x_test_unvec = subset_reencode_features(
                x_test_unvec, feat_encoding_dict)
            num_feature = max_num_feature

        # interpret
        start = time.time()

        temp_mlp = MLP(num_feature=num_feature, num_class=num_class)
        hdf5_path = full_out_dir + '/model.h5'
        sums_D, sums_D2, sums_contribs, pairs = \
            feature_importance.get_diff_sums(
                hdf5_path,
                x_test_unvec,
                process_x_func=temp_mlp.process_x,
                num_feature=num_feature,
                num_class=num_class)

        with open(full_out_dir + '/sums_D.pkl', 'wb') as f:
            pickle.dump(sums_D, f)
        with open(full_out_dir + '/sums_D2.pkl', 'wb') as f:
            pickle.dump(sums_D2, f)
        with open(full_out_dir + '/sums_contribs.pkl', 'wb') as f:
            pickle.dump(sums_contribs, f)

        list_sums_D.append(sums_D)
        list_sums_D2.append(sums_D2)
        list_sums_contribs.append(sums_contribs)

    def compute_total_sums(list_sums):
        total_sums = list_sums[0]

        for i in range(1, len(list_sums)):
            for j in range(len(total_sums)):
                total_sums[j] = np.add(total_sums[j], list_sums[i][j])

        return total_sums

    total_sums_D = compute_total_sums(list_sums_D)
    total_sums_D2 = compute_total_sums(list_sums_D2)
    total_sums_contribs = compute_total_sums(list_sums_contribs)

    num_sample = len(x_unvec)
    run_interpretation_summary(
        x_unvec, y, total_sums_D, total_sums_D2, total_sums_contribs,
        idx_feat_dict=idx_feat_dict, idx_class_dict=idx_class_dict,
        icd9_descript_dict=icd9_descript_dict, pairs=pairs,
        num_sample=num_sample, full_out_dir=base_out_dir)

    print('Computed DeepLIFT scores and analysis in {:.4f} seconds'
          .format(time.time() - start))
    print('-' * 72)
    print()


def main():
    """Main method."""
    np.random.seed(SEED)  # for reproducibility, must be before Keras imports!
    run(data_fn=FLAGS.data_fn,
        prop_missing=FLAGS.prop_missing,
        max_num_feature=FLAGS.max_num_feature,
        feature_selection=FLAGS.feature_selection,
        data_dir=FLAGS.data_dir,
        out_dir=FLAGS.out_dir)


# if run as script, execute main
if __name__ == '__main__':
    FLAGS, _ = parser.parse_known_args()
    main()
