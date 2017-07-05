"""
frequency.py

Counts occurences of features for each class. 

Requires:   NumPy (and its dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

import numpy as np

'''
* Creates a frequency table counting the number of occurences of each feature 
  (corresponding to column) for each class (corresponding to row). The table 
  is represented as as numpy array with dimensions (nb_classes, nb_features).
* Expects:
    - X = feature data as a list of list of features
        e.g., [['30', 'M'. '311:30', 'V72.3:4'], ['55', 'F' '311:18']]
    - y = list of true classes
        e.g., ['H', 'O']
    - idx_feat_dict = dictionary mapping feature indices to feature names
    - idx_class_dict = dictionary mapping class indices to class names
* Returns:
    - frequency table, as described above
'''
def get_frequency_table(X, y, idx_feat_dict, idx_class_dict):
    nb_features = len(idx_feat_dict)
    nb_classes = len(idx_class_dict)

    # initialize table with zeros
    class_feat_freq_table = np.zeros((nb_classes, nb_features))

    for features, class_idx in zip(X, y):
        for feat_idx in features:
            class_feat_freq_table[class_idx][feat_idx] += 1

    # check ordering of features (should be most popular = lowest index)
    rowsums = np.sum(class_feat_freq_table, axis=0)
    for idx in range(nb_features - 1):
        assert rowsums[idx] >= rowsums[idx + 1]
    
    return class_feat_freq_table