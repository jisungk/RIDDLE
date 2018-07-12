"""ordering.py

Creates orderings of classes based on various scores (feature contribution
scores, feature frequency, proportion of samples exhbiting a feature).

Requires:   NumPy (and its dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

from __future__ import print_function

from collections import OrderedDict
import pickle

import numpy as np

from .summary import Summary
from . import feature_importance

# how to handle floating pt errs
np.seterr(divide='ignore', over='raise', under='raise')

PRECISION = 4


class OrderingSummary(Summary):
    """Summary of ordering information."""

    def sort(self):
        """TODO(jisungk): implement and write docstring."""
        raise NotImplementedError()
        # def get_sorted_indices(od):
        #     list_to_sort = od['contrib_ordering']
        #     # first subset index = get item and not enumerated index
        #     # second subset index = get a particular (class, value) pair
        #     # third subset index = get value and not class name
        #     # => sorts by range (e.g., max - min) of contribution values
        #     # assumes (class, value) pairs are sorted already for each feature
        #     sorted_indices = [i[0] for i in sorted(
        #         enumerate(list_to_sort), key=lambda x: x[1][0][1] - x[1][-1][1],
        #         reverse=True)]
        #     return sorted_indices

        # sorted_indices = get_sorted_indices(self.od)
        # # sorts each value accordingt to the list of sorted indices
        # for key, value in self.od.items():
        #     sorted_value = [value[i] for i in sorted_indices]
        #     self.od[key] = sorted_value

    def save(self, out_dir):
        """Saves ordering summary dictionary and CSV table to file.

        There are 5 columns:
            feature, e.g., V02.00
            descript, e.g., pneumonia
            contrib_ordering, e.g., (O, 150) > (H, 100) > (B, -50) > (W, -50)
                This is ordering based on the sum of contribution scores
            freq_ordering, e.g., (H, 42) > (O, 10) > (W, 4) > (B, 1)
                This is ordering based on the frequency of features per ethnicity.
            prop_ordering, e.g., (O, 0.9) > (H, 0.6) > (B, 0.5) > (W, 0.1)
                This is ordering based on the proportion of the ethnic group who
                who have the relevant feature.

        Arguments:
            out_dir: string
                directory where the outputs should be saved
        """
        ordering_keys = [key for key in self.od.keys() if 'ordering' in key]

        with open(out_dir + '/orderings_ordered_dict.pkl', 'wb') as f:
            pickle.dump(self.od, f)

        with open(out_dir + '/orderings.txt', 'w') as f:
            # write header
            [f.write(str(key) + '\t') for key in list(self.od.keys())[:-1]]
            f.write(str(list(self.od.keys())[-1]) + '\n')

            num_feature = len(list(self.od.values())[0])
            num_columns = len(self.od.keys())

            for feat_idx in range(num_feature):
                for col_idx, (key, value) in enumerate(self.od.items()):
                    def _order_pair_to_string(pair, precision=PRECISION):
                        first = pair[0]
                        second = pair[1]
                        if isinstance(second, float):  # if float value, round
                            second = round(second, precision)

                        return str(first) + ' (' + str(second) + ')'

                    v = value[feat_idx]
                    if key in ordering_keys:
                        v = ' > '.join([_order_pair_to_string(p) for p in v])
                    assert '\t' not in v  # would screw up tab-delim

                    f.write(str(v))
                    if col_idx < num_columns - 1:
                        f.write('\t')
                    else:
                        f.write('\n')

    def save_individual_tables(self, idx_class_dict, out_dir):
        """Save CSV tables describing class-specific scores across features.

        The number of tables depends on the number of orderings (e.g., DeepLIFT
        score, frequency, proportion). There are num_class + 1 columns:
            feature, e.g., V02.00
            score for first class (0)
            score for second class (1)
            ...
            score for last class (num_class - 1)

        Arguments:
            idx_class_dict: {int: string}
                dictionary mapping class indices to classes
            out_dir: string
                directory where the outputs should be saved
        """
        class_idx_dict = {val: key for key, val in idx_class_dict.items()}
        num_class = len(idx_class_dict)

        ordering_keys = [key for key in self.od.keys() if 'ordering' in key]
        feat_key = 'feat'

        for key in ordering_keys:
            curr_table_data = self.od[key]
            with open(out_dir + '/' + key + '_table.txt', 'w') as f:
                # write header
                f.write(feat_key)
                f.write('\t')
                for class_idx in range(num_class):
                    f.write(idx_class_dict[class_idx])
                    if class_idx < num_class - 1:
                        f.write('\t')
                    else:
                        f.write('\n')

                for row_idx, list_pairs in enumerate(curr_table_data):
                    f.write(self.od[feat_key][row_idx])  # write feat
                    f.write('\t')

                    scores = [''] * num_class
                    for class_value_pair in list_pairs:
                        class_idx = class_idx_dict[class_value_pair[0]]
                        assert scores[class_idx] == ''  # shouldn't be init.
                        score = str(class_value_pair[1])
                        assert '\t' not in score  # would screw up tab-delim
                        scores[class_idx] = score

                    assert all([s != '' for s in scores])
                    f.write('\t'.join(scores))
                    f.write('\n')


def summarize_orderings(contrib_sums, class_feat_freq_table, idx_feat_dict,
                        idx_class_dict, icd9_descript_dict):
    """Creates an OrderingSummary object using various orderings.

    Arguments:
        contrib_sums: np.ndarray, float
            2-D array of sums of DeepLIFT contribution scores with shape
            (num_class, num_feature); the outer (0) dim represents the class and
            the inner dim (1) represents the sum of scores across features
        class_feat_freq_table: np.ndarray, int
            2-D array of frequencies with shape (num_class, num_feature); the
            outer (0) dim represents the class and the inner dim (1) represents
            the frequency value across features
        idx_feat_dict: {int: string}
            dictionary mapping feature indices to features
        idx_class_dict: {int: string}
            dictionary mapping class indices to classes
        icd9_descript_dict: {string: string}
            dictionary mapping ICD9 codes to description text

    Returns:
        summary: OrderingSummary
            OrderingSummary object representing a summary of orderings
    """
    class_feat_prop_table = _compute_prop_table(class_feat_freq_table)

    list_contrib_orderings = _decode_orderings(
        _compute_orderings(contrib_sums), idx_class_dict)
    list_freq_orderings = _decode_orderings(
        _compute_orderings(class_feat_freq_table), idx_class_dict)
    list_prop_orderings = _decode_orderings(
        _compute_orderings(class_feat_prop_table), idx_class_dict)

    list_feat_names = feature_importance._get_list_feat_names(
        idx_feat_dict, num_pair=1)  # since we only want 1 repetition
    list_feat_descripts = feature_importance._get_list_feat_descripts(
        list_feat_names, icd9_descript_dict=icd9_descript_dict)

    summary = OrderingSummary(OrderedDict(
        [('feat', list_feat_names),
         ('descript', list_feat_descripts),
         ('contrib_ordering', list_contrib_orderings),
         ('freq_ordering', list_freq_orderings),
         ('prop_ordering', list_prop_orderings)]))

    return summary


def _compute_orderings(score_array):
    """Compute orderings of classes given a 2D array of scores.

    Orders from most postivei to most negative.

    Arguments:
        score_array: np.ndarray, float or int
            2-D array of scores with shape (num_class, num_feature); the
            outer (0) dim represents the class and the inner dim (1) represents
            the score across features

    Returns:
        orderings: [[(int, float or int)]]
            list of list of pairs representing the orderings of classes
            (inner sublist) across features (outer list); in the inner pair
            the first value represents the class index and the second value
            represents the score
    """
    num_class = score_array.shape[0]
    num_feature = score_array.shape[1]

    orderings = []
    for feat_idx in range(num_feature):
        # list of (class index, class-specific score) for curr feature
        class_scores = [(class_idx, score_array[class_idx][feat_idx])
                        for class_idx in range(num_class)]
        ordered_classes = sorted(
            class_scores, key=lambda x: x[1], reverse=True)
        orderings.append(ordered_classes)

    return orderings


def _compute_prop_table(class_feat_freq_table):
    """Creates a frequency proportion table from a frequency table.

    The first dimension (0) corresponds to classes and the second dimension (1)
    corresponds to features. A proportion score is defined as the proportion
    of samples within a given class which exhibit a given feature

    Arguments:
        class_feat_freq_table: np.ndarray, int
            2-D array of frequencies with shape (num_class, num_feature); the
            outer (0) dim represents the class and the inner dim (1) represents
            the frequency value across features

    Returns:
        prop_feat_freq_table: np.ndarray, float
            2-D array of frequency proportions with shape
            (num_class, num_feature); the outer (0) dim represents the class and
            the inner dim (1) represents the proportion value across features
    """
    num_feature = class_feat_freq_table.shape[1]

    rowsums = np.sum(class_feat_freq_table, axis=1)
    concat_rowsums = np.transpose(np.repeat([rowsums], num_feature, axis=0))
    assert concat_rowsums.shape == class_feat_freq_table.shape

    # important to ensure that numpy array is of type float to prevent
    # integer division
    return np.true_divide(class_feat_freq_table, concat_rowsums)


def _decode_orderings(raw_orderings, idx_class_dict):
    """Converts encoded orderings into orderings with original class names.

    Arguments:
        raw_orderings: [[(int, float or int)]]
            list of list of pairs representing the orderings of classes
            (inner sublist) across features (outer list); in the inner pair
            the first value represents the class index and the second value
            represents the score
        idx_class_dict: {int: string}
            dictionary mapping class indices to classes

    Returns:
        decoded_orderings: [[(string, float or int)]]
            list of list of pairs representing the orderings of classes
            (inner sublist) across features (outer list); in the inner pair
            the first value represents the class name and the second value
            represents the score
        idx_class_dict: {int: string}
            dictionary mapping class indices to classes
    """
    decoded_orderings = [[(idx_class_dict[pair[0]], pair[1]) for pair in o]
                         for o in raw_orderings]
    return decoded_orderings
