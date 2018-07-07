"""frequency.py

Counts occurences of features for each class.

Requires:   NumPy (and its dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

import numpy as np


def get_frequency_table(x_unvec, y, idx_feat_dict, idx_class_dict):
    """Creates a frequency table counting occurences of features across classes.

    The first dimension (0) corresponds to classes and the second dimension (1)
    corresponds to features.

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

    Returns:
        class_feat_freq_table: np.ndarray, int
            2-D array of frequencies with shape (num_class, num_feature); the
            outer (0) dim represents the class and the inner dim (1) represents
            the frequency value across features
    """
    num_feature = len(idx_feat_dict)
    num_class = len(idx_class_dict)

    # initialize table with zeros
    class_feat_freq_table = np.zeros((num_class, num_feature))

    for features, class_idx in zip(x_unvec, y):
        for feat_idx in features:
            class_feat_freq_table[class_idx][feat_idx] += 1

    # check ordering of features (should be most popular = lowest index)
    # not true if used on holdout data, only true if used on all data
    # rowsums = np.sum(class_feat_freq_table, axis=0)
    # for idx in range(num_feature - 1):
    #     assert rowsums[idx] >= rowsums[idx + 1]

    return class_feat_freq_table
