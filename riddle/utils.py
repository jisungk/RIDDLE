"""
utils.py

Provides useful functions.

Requires:   NumPy (and its dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

from math import ceil

import numpy as np

# -------------------------------- FUNCTIONS --------------------------------- #

'''
* Prints to standard error.
'''
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

'''
* Randomly & coordinately splits two indexable items, in accordance with
  k-fold cross-validation parameters.
* Expects:
    - X = indexable item (typically list of samples)
    - y = indexable item (typically list of classes)
    - k_idx = the index of the partition to use
    - k = the # of partitions for k-fold cross-validation
* Returns:
    - duple representing the larger (k - 1)/k portion of split X, y data
    - duple representing the smaller 1/k portion of split X, y data
'''
def split_data(X, y, k_idx, k, perm_indices):
    assert k > 0
    assert k_idx >= 0
    assert k_idx < k

    N = len(X)
    partition_size = int(ceil(N / k))

    # minority group is the single selected partition
    # majority group is the other partitions
    minority_start = k_idx * partition_size
    minority_end = minority_start + partition_size
    minority_indices = perm_indices[minority_start:minority_end]

    majority_indices = np.append(perm_indices[0:minority_start], 
        perm_indices[minority_end:])

    X_majority = [X[i] for i in majority_indices]
    y_majority = [y[i] for i in majority_indices]
    X_minority  = [X[i] for i in minority_indices]
    y_minority  = [y[i] for i in minority_indices]

    assert len(X_majority) + len(X_minority) == N

    return (X_majority, y_majority), (X_minority, y_minority)
