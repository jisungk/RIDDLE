"""emr.py

Preprocesses EMR data files. Files should have 1 case per line and be space
delimited; each case begins with the age, ethnicity class, gender, then
subsequent ICD9 codes. Does not perform vectorization (e.g., one-hot-encoding).

Author:     Ji-Sung Kim (Rzhetksy Lab)
Copyright:  2018, all rights reserved
"""

from __future__ import print_function

from collections import Counter
from math import ceil

import numpy as np

# column information
RAW_AGE_COL, RAW_CLASS_COL, RAW_GENDER_COL, RAW_FIRST_ICD9_COL = 0, 1, 2, 3


def get_data(path, icd9_descript_dict, no_onset_age=True):
    """Get EMR data in standard (x, y) format (does not perform vectorization).

    Arguments:
        path: string
            file filepath
        icd9_descript_dict: {string: string}
            dictionary mapping ICD9 codes to description text
        no_onset_age: bool
            whether to discard onset_age information

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
    """
    data = _read_file(path)
    x_raw, y_raw = _clean_data(data, icd9_descript_dict=icd9_descript_dict,
                               no_onset_age=no_onset_age)

    (feat_idx_dict, idx_feat_dict, class_idx_dict,
     idx_class_dict) = _get_dicts(x_raw, y_raw)

    x_unvec, y = _encode(x_raw, y_raw, feat_idx_dict, class_idx_dict)

    del x_raw
    del y_raw

    return x_unvec, y, idx_feat_dict, idx_class_dict


def get_k_fold_partition(x_unvec, y, k_idx, k, perm_indices):
    """Splits data into training, validation, and testing sets.

    Splits data in accordance with on k-fold cross-validation.

    Arguments:
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        y: [int]
            list of class labels as integer indices
        idx_feat_dict: {int: string}
        k_idx: int
            index of the k-fold partition to use
        k: int
            number of partitions for k-fold cross-validation
        perm_indices: np.ndarray, int
            array of indices representing a permutation of the samples with
            shape (num_sample, )

    Returns:
        x_unvec_train: [[int]]
            training feature indices that have not been vectorized; each inner
            list collects the indices of features that are present (binary on)
            for a sample
        y_train: [int]
            list of training class labels as integer indices
        x_unvec_val: [[int]]
            validation feature indices that have not been vectorized; each inner
            list collects the indices of features that are present (binary on)
            for a sample
        y_tval: [int]
            list of validation class labels as integer indices
        x_unvec_test: [[int]]
            test feature indices that have not been vectorized; each inner
            list collects the indices of features that are present (binary on)
            for a sample
        y_test: [int]
            list of test class labels as integer indices
    """
    (x_unvec_train, y_train), (x_unvec_test, y_test) = _split_data(
        x_unvec, y, k_idx=k_idx, k=k, perm_indices=perm_indices)

    # typically, 10% of training data => validation (so k = 10)\
    val_perm_indices = perm_indices[perm_indices < len(x_unvec_train)]
    (x_unvec_train, y_train), (x_unvec_val, y_val) = _split_data(
        x_unvec_train, y_train, k_idx=0, k=10, perm_indices=val_perm_indices)

    assert len(x_unvec_train) + len(x_unvec_val) + \
        len(x_unvec_test) == len(x_unvec)

    return x_unvec_train, y_train, x_unvec_val, y_val, x_unvec_test, y_test


def get_icd9_descript_dict(path):
    """Read from file a dictionary mapping ICD9 codes to their descriptions.

    Arguments:
        path: string
            file filepath

    Returns:
        icd9_descript_dict: {string: string}
            dictionary mapping ICD9 codes to description text
    """
    lines = _read_file(path)
    icd9_descript_dict = {}

    for l in lines[1:]:  # ignore first line which is column names
        elems = l.split('\t')

        try:
            assert len(elems) == 8  # number of columns should be 8
        except:
            print('Problem with following line while loading icd9_descript_dict:')
            print(l)
            raise

        icd9 = elems[0]  # ICD9 code should be in the first column
        descript = elems[1]  # description should be in the second column

        # check if the ICD9 code is a category and if so, append a label
        is_category = len(icd9.split('.')) == 1
        if is_category:
            descript = descript + ' (category)'

        icd9_descript_dict[icd9] = descript

    return icd9_descript_dict


def _read_file(path):
    """Read lines from a file.

    Arguments:
        path: string
            file filepath

    Returns:
        data: [string]
            list of string lines from file
    """
    with open(path) as f:
        data = f.read().splitlines()
    return data


def _clean_data(data, icd9_descript_dict, no_onset_age=True):
    """Clean up and annotates data; converts data into standard (x, y) form.

    Arguments:
        data: [string]
            list of string lines, e.g., ['8 H M 31:8 V72.3:4', '55 O F 31:18']
        icd9_descript_dict: {string: string}
            dictionary mapping ICD9 codes to description text
        no_onset_age: bool
            whether to discard onset_age information

    Returns:
        x_raw: [[string]
            list of list of string features; outer list represents samples
            e.g., [['age_8', 'gender_M', '31', 'V72.3'],
                   ['age_55', 'gender_F', '31']]
        y_raw: [string]
            list of classes
    """
    x_raw, y_raw = [], []

    for idx, line in enumerate(data):
        line = line.split()

        try:
            features = []
            features.append('age_' + line[RAW_AGE_COL])
            features.append('gender_' + line[RAW_GENDER_COL])

            icd9s = [i.split(':') for i in line[RAW_FIRST_ICD9_COL:]]
            # filter invalid icd9s and sort by onset age in place
            icd9s = [i for i in icd9s if i[0] in icd9_descript_dict]
            icd9s.sort(key=lambda i: int(i[1]))

            if no_onset_age:
                icd9s = [i[0] for i in icd9s]  # remove onset age
            else:
                icd9s = [':'.join(i) for i in icd9s]
            features.extend(icd9s)

            x_raw.append(features)
            y_raw.append(line[RAW_CLASS_COL])  # extract class
        except:
            print('WARNING: error on line #{} with case:'.format(idx))
            print(' '.join(line))
            raise

    assert len(x_raw) == len(y_raw)

    return x_raw, y_raw


def _get_dicts(x_raw, y_raw):
    """Map features and classes to integer values, "indices".

    Arguments:
        x_raw: [[string]
            list of list of string features; outer list represents samples
            e.g., [['age_8', 'gender_M', '31', 'V72.3'],
                   ['age_55', 'gender_F', '31']]
        y_raw: [string]
            list of classes

    Returns:
        feat_idx_dict: {string: int}
            dictionary mapping features to feature indices
        idx_feat_dict: {int: string}
            dictionary mapping feature indices to features
        class_idx_dict: {string: int}
            dictionary mapping classes to class indices
        idx_class_dict: {int: string}
            dictionary mapping class indices to classes
    """
    feat_counts = Counter([f for line in x_raw for f in line])
    class_counts = Counter(y_raw)

    feat_idx_dict, idx_feat_dict, class_idx_dict, idx_class_dict = {}, {}, {}, {}

    for idx, (c, _) in enumerate(class_counts.most_common()):
        class_idx_dict[c] = idx
        idx_class_dict[idx] = c

    for idx, (feat, _) in enumerate(feat_counts.most_common()):
        feat_idx_dict[feat] = idx
        idx_feat_dict[idx] = feat

    return feat_idx_dict, idx_feat_dict, class_idx_dict, idx_class_dict


def _encode(x_raw, y_raw, feat_idx_dict, class_idx_dict):
    """Encode features and classes to integer indices using given dictionaries.

    Arguments:
        x_raw: [[string]
            list of list of string features; outer list represents samples
            e.g., [['age_8', 'gender_M', '31', 'V72.3'],
                   ['age_55', 'gender_F', '31']]
        y_raw: [string]
            list of classes
        feat_idx_dict: {string: int}
            dictionary mapping features to feature indices
        class_idx_dict: {string: int}
            dictionary mapping classes to class indices

    Returns:
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        y: [int]
            list of class labels as integer indices
    """
    x_unvec = [[feat_idx_dict[feat] for feat in line] for line in x_raw]
    y = [class_idx_dict[c] for c in y_raw]
    assert len(x_unvec) == len(y)

    return x_unvec, y


def _split_data(x, y, k_idx, k, perm_indices):
    """Randomly and coordinates splits two indexable items.

    Splits items in accordiance with k-fold cross-validatoin.

    Arguments:
        x: [?]
            indexable item
        y: [?]
            indexable item
        k_idx: int
            index of the k-fold partition to use
        k: int
            number of partitions for k-fold cross-validation
        perm_indices: np.ndarray, int
            array of indices representing a permutation of the samples with
            shape (num_sample, )
    Returns:
        x_majority: [?]
            majority partition of indexable item
        y_majority: [?]
            majority partition of indexable item
        x_minority: [?]
            minority partition of indexable item
        y_minority: [?]
            minority partition of indexable item
    """
    assert k > 0
    assert k_idx >= 0
    assert k_idx < k

    N = len(x)
    partition_size = int(ceil(N / k))

    # minority group is the single selected partition
    # majority group is the other partitions
    minority_start = k_idx * partition_size
    minority_end = minority_start + partition_size

    minority_indices = perm_indices[minority_start:minority_end]
    majority_indices = np.append(perm_indices[0:minority_start],
                                 perm_indices[minority_end:])

    assert np.array_equal(np.sort(np.append(minority_indices, majority_indices)),
                          np.array(range(N)))

    x_majority = [x[i] for i in majority_indices]
    y_majority = [y[i] for i in majority_indices]
    x_minority = [x[i] for i in minority_indices]
    y_minority = [y[i] for i in minority_indices]

    return (x_majority, y_majority), (x_minority, y_minority)
