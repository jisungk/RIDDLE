"""
emr.py

Preprocesses EMR data files. Files should have 1 case per line and be space 
delimited; each case begins with the age, ethnicity class, gender, then 
subsequent ICD9 codes. Does not perform vectorization (e.g., one-hot-encoding).

Author: Ji-Sung Kim (Rzhetksy Lab)
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

from collections import Counter

from . import utils
import numpy as np

# --------------------------------- SETTINGS --------------------------------- #

# column information
RAW_AGE_COL, RAW_CLASS_COL, RAW_GENDER_COL, RAW_FIRST_ICD9_COL = 0, 1, 2, 3
X_AGE_COL, X_GENDER_COL, X_FIRST_ICD9_COL = 0, 1, 2

# ---------------------------- HELPER FUNCTIONS ------------------------------ #

'''
* Reads in a file.
* Expects:
    - path = string filepath of the file to be loaded
* Returns:
    - list of string lines from file. 
'''
def read_file(path):
    with open(path) as f:
        data = f.read().splitlines()
    return data

'''
* Cleans up and annotates data; converts data into standard (X, y) form.
* Expects:
    - data = list of string lines
        e.g., ['8 H M 31:8 V72.3:4', '55 O F 31:18']
    - icd9_descript_dict = dictionary mapping ICD9 codes to their descriptions
    - no_onset_age = boolean for whether to discard onset age information 
* Returns:
    - X which is a list of list of features; outer list represents samples
        e.g., ['age_8', 'gender_M', '31', 'V72.3'], ['age_55', 'gender_F', '31']]
    - y which is a list of true classes
        e.g., ['H', 'O']
'''
def clean_data(data, icd9_descript_dict, no_onset_age=True):
    X, y = [], []

    count = 0
    for idx, line in enumerate(data):
        line = line.split()

        try:
            features = []
            features.append('age_' + line[RAW_AGE_COL])
            features.append('gender_' + line[RAW_GENDER_COL])

            icd9s = [i.split(':') for i in line[RAW_FIRST_ICD9_COL:]]
            # filter invalid icd9s and sort by onset age in place
            icd9s = [i for i in icd9s if i[0] in icd9_descript_dict]
            icd9s.sort(key = lambda i: int(i[1]))

            if no_onset_age: icd9s = [i[0] for i in icd9s] # remove onset age
            else: icd9s = [':'.join(i) for i in icd9s]
            features.extend(icd9s)

            X.append(features)
            y.append(line[RAW_CLASS_COL]) # extract class
        except:
            print('WARNING: error on line #{} with case:'.format(idx))
            print(' '.join(line))
            raise
    
    assert len(X) == len(y)

    return X, y


''' 
* Randomly removes some proportion of feature data for the purpose of 
  simulating randomly missing data.
* Expects:
    - X = list of list of features; outer list represents samples
        e.g., ['age_8', 'gender_M', '31', 'V72.3'], ['age_55', 'gender_F', '31']]
    - prop = float for the proportion of data to be removed
* Returns:
    - X which is a list of list of features
        e.g., ['age_8', 'gender_M', '31'], ['age_55', '31']]
'''
def simulate_missing_data(X, prop):
    print(X[0])
    if prop < 0:
        raise ValueError('prop for simulating missing data is negative')
    if prop == 0.0: return X

    indices = [[(row_idx, col_idx) for col_idx in range(len(x))] \
        for row_idx, x in enumerate(X)]
    indices = [i for r in indices for i in r] # flatten
    np.random.shuffle(indices)

    nb_to_remove = int(len(indices) * prop)
    indices_to_remove = indices[:int(len(indices) * prop)]
    # need to sort these so that deletion works and doesn't screw up indices
    indices_to_remove = sorted(indices_to_remove, key=lambda x: (x[0], x[1]),
        reverse=True)

    for row_idx, col_idx in indices_to_remove:
        X[row_idx].pop(col_idx)

    # check number of removals
    lengths = [len(x) for x in X]
    assert sum(lengths) == len(indices) - nb_to_remove

    print('Randomly deleted {} feature occurrences'.format(nb_to_remove))

    return X

'''
* Maps features and classes to integer values, "indices". More frequent values
  are assigned lower indices. 
* Expects:
    - X = list of list of features; outer list represents samples
        e.g., ['age_8', 'gender_M', '31', 'V72.3'], ['age_55', 'gender_F', '31']]
    - y = list of true classes
        e.g., ['H', 'O']
* Returns:
    - dictionary mapping feature names to feature indices
    - dictionary mapping feature indices to feature names
    - dictionary mapping class names to class indices
    - dictionary mapping class indices to class names
'''
def get_dicts(X, y):
    feat_counts = Counter()
    class_counts = Counter(y)

    def count_feat(f):
        feat_counts[f] += 1
    map(count_feat, [f for line in X for f in line])

    feat_idx_dict, idx_feat_dict, class_idx_dict, idx_class_dict = {}, {}, {}, {}

    for idx, (c, count) in enumerate(class_counts.most_common()):
        class_idx_dict[c] = idx
        idx_class_dict[idx] = c

    for idx, (feat, count) in enumerate(feat_counts.most_common()):
        feat_idx_dict[feat] = idx 
        idx_feat_dict[idx] = feat

    return feat_idx_dict, idx_feat_dict, class_idx_dict, idx_class_dict

'''
* Encodes features and classes to integer indices, based on given dictionaries.
* Expects:
    - X = list of list of features; outer list represents samples
        e.g., ['age_8', 'gender_M', '31', 'V72.3'], ['age_55', 'gender_F', '31']]
    - y = list of true classes
        e.g., ['H', 'O']
    - feat_idx_dict = dictionary mapping feature names to feature indices
    - class_idx_dict = dictionary mapping class names to class indices
* Returns:
    - encoded X which is a list of list of feature indices
        e.g., [[0, 3, 1, 2], [4, 5, 1]]
    - encoded y which is a list of list of class indices
        e.g., [0, 1]
'''
def encode(X, y, feat_idx_dict, class_idx_dict):
    X = [[feat_idx_dict[feat] for feat in line] for line in X]
    y = [class_idx_dict[c] for c in y]
    assert len(X) == len(y)

    return X, y

'''
* Gets EMR data in standard (features, target) format. Does not perform 
  vectorization.
* Expects:
    - path = string filepath of the file to be loaded
    - icd9s_only = boolean for whether to keep only ICD9 codes as features
    - no_onset_age = boolean for whether to discard onset age information
    - prop_missing = float for the proportion of data to be removed
* Returns:
    - X which is the feature data as a list of list of feature indices
    - y which is the class data as a list of list of class indices
    - dictionary mapping feature indices to feature names
    - dictionary mapping class indices to class names
'''
def get_data(path, icd9_descript_dict, icd9s_only=False, 
    no_onset_age=True, prop_missing=0.0):
    data = read_file(path)
    X, y = clean_data(data, icd9_descript_dict=icd9_descript_dict, 
        no_onset_age=no_onset_age)

    if prop_missing != 0.0:
        X = simulate_missing_data(X, prop=prop_missing)
        
    feat_idx_dict, idx_feat_dict, class_idx_dict, idx_class_dict = \
        get_dicts(X, y)

    X, y = encode(X, y, feat_idx_dict, class_idx_dict)
    return X, y, idx_feat_dict, idx_class_dict

# ---------------------------- PUBLIC FUNCTIONS ------------------------------ #

'''
* Splits data into training, validation, and testing sets based on k-fold 
  partitioning. 
* Expects:
    - X = feature data as a list of list of feature indices
        e.g., [[0, 3, 1, 2], [4, 5, 1]]
    - y = class data as a list of list of class indices
        e.g., [0, 1]
    - k_idx = index of the selected partition
    - k = number of partitions
    - perm_indices = list of randomly shuffled indices from 0 to nb_samples
        e.g., [3, 1, 2, 0]
* Returns:
    - data partition as a dictionary with 6 keys for each subset: 'X_train', 
      'y_train', 'X_val, 'y_val', 'X_test', 'y_test'
'''
def get_k_fold_partition(X, y, k_idx, k, perm_indices):
    (X_train, y_train), (X_test, y_test) = utils.split_data(X, y, k_idx=k_idx, 
        k=k, perm_indices=perm_indices) 

    # typically, 10% of training data => validation (so k = 10)\
    val_perm_indices = perm_indices[perm_indices < len(X_train)]
    (X_train, y_train), (X_val, y_val) = utils.split_data(X_train, y_train,
        k_idx=0, k=10, perm_indices=val_perm_indices) 

    assert len(X_train) + len(X_val) + len(X_test) == len(X)

    return {'X_train': X_train, 'y_train':y_train, 'X_val':X_val, 'y_val':y_val, 
        'X_test':X_test, 'y_test':y_test}

'''
* Creates a dictionary mapping ICD9 codes to their descriptions.
* Expects:
    - path = string filepath of the file to be loaded
* Returns:
    - dictionary mapping ICD9 codes to their descriptions
'''
def get_icd9_descript_dict(path):
    lines = read_file(path)
    icd9_descript_dict = {}

    for l in lines[1:]: # ignore first line which is column names
        elems = l.split('\t')
       
        try: 
            assert len(elems) == 8 # number of columns should be 8
        except:
            print('Problem with following line while loading icd9_descript_dict:')
            print(l)
            raise

        icd9 = elems[0] # ICD9 code should be in the first column
        descript = elems[1] # description should be in the second column

        # check if the ICD9 code is a category and if so, append a label
        is_category = len(icd9.split('.')) == 1
        if is_category: descript = descript + ' (category)'

        icd9_descript_dict[icd9] = descript 

    return icd9_descript_dict