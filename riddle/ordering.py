"""
ordering.py

Creates orderings of classes based on various scores (feature contribution
scores, feature frequency, proportion of samples exhbiting a feature).

Requires:   NumPy (and its dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

import sys
import math
import pickle
from collections import OrderedDict

import numpy as np

from .summary import Summary
from . import feature_importance

# -------------------------------- SETTINGS ---------------------------------- #

# how to handle floating pt errs
np.seterr(divide='ignore', over='raise', under='raise')

# ----------------------------- HELPER CLASSES ------------------------------- #

class OrderingSummary(Summary):
    def sort(self):
        '''
        * Gets a list of sorted indices by descending range of contribution 
          scores.
        * Expects: 
            - od = OrderedDictionary mapping keys to lists of values
        * Returns:
            - a list of sorted indices
                e.g., [0, 3, 2, 1]
        '''
        def get_sorted_indices(od):
            list_to_sort = od['contrib_ordering']
            # first subset index = get item and not enumerated index
            # second subset index = get a particular (class, value) pair
            # third subset index = get value and not class name
            # => sorts by range (e.g., max - min) of contribution values
            # assumes (class, value) pairs are sorted already for each feature
            sorted_indices = [i[0] for i in sorted(enumerate(list_to_sort), 
                key=lambda x:x[1][0][1] - x[1][-1][1], reverse=True)]
            return sorted_indices

        sorted_indices = get_sorted_indices(self.od)
        # sorts each value accordingt to the list of sorted indices
        for key, value in self.od.items():
            sorted_value = [value[i] for i in sorted_indices]
            self.od[key] = sorted_value

    '''
    * Saves to file information a tab-delimited table summarizing the ordering
      of classes by features based on contribution scores. There are 5 columns:
        + feature, e.g., V02.00
        + descript, e.g., pneumonia
        + contrib_ordering, e.g., (O, 150) > (H, 100) > (B, -50) > (W, -50)
            * This is ordering based on the sum of contribution scores
        + freq_ordering, e.g., (H, 42) > (O, 10) > (W, 4) > (B, 1)
            * This is ordering based on the frequency of features per ethnicity.
        + prop_ordering, e.g., (O, 0.9) > (H, 0.6) > (B, 0.5) > (W, 0.1)
            * This is ordering based on the proportion of the ethnic group who
              who have the relevant feature.
    * Expects:
        - out_directory = string path describing where to save files
            e.g., "my_folder" 
    '''
    def save(self, out_directory):
        ordering_keys = [key for key in self.od.keys() if 'ordering' in key]

        with open(out_directory + '/orderings_ordered_dict.pkl', 'w') as f:
            pickle.dump(self.od, f)

        with open(out_directory + '/orderings.txt', 'w') as f:
            # write header
            [f.write(str(key) + '\t') for key in self.od.keys()[:-1]]
            f.write(str(self.od.keys()[-1]) + '\n')

            nb_features = len(self.od.values()[0])
            nb_columns = len(self.od.keys())

            for feat_idx in range(nb_features):
                for col_idx, (key, value) in enumerate(self.od.items()):
                    def order_pair_to_string(pair, precision=4):
                        first = pair[0]
                        second = pair[1]
                        if isinstance(second, float): # if float value, round
                            second = round(second, precision)

                        return str(first) + ' (' + str(second) + ')' 
                    
                    v = value[feat_idx]
                    if key in ordering_keys: 
                        v = ' > '.join([order_pair_to_string(p) for p in v])
                    assert '\t' not in v # would screw up tab-delim

                    f.write(str(v))
                    if col_idx < nb_columns - 1: f.write('\t')
                    else: f.write('\n')
    '''
    * Saves to file CSV tables describing the class-specific score for each
      feature. Individual tables are created for each type of score used
      for ordering (e.g., contribution score, frequency, proportion). 
      There are nb_classes + 1 columns:
        + feature, e.g., V02.00
        + [first class]
        + [second class]
        + ...
        + [last class]
    * Expects:
        - idx_class_dict = dictionary mapping class indices to class names
        - out_directory = string path describing where to save files
            e.g., "my_folder" 
    '''
    def save_individual_tables(self, idx_class_dict, out_directory):        
        class_idx_dict = {val: key for key, val in idx_class_dict.items()}
        nb_classes = len(idx_class_dict)

        ordering_keys = [key for key in self.od.keys() if 'ordering' in key]
        feat_key = 'feat'

        for key in ordering_keys:
            curr_table_data = self.od[key]
            with open(out_directory + '/' + key + '_table.txt', 'w') as f:
                # write header
                f.write(feat_key)
                f.write('\t')
                for class_idx in range(nb_classes):
                    f.write(idx_class_dict[class_idx])
                    if class_idx < nb_classes - 1: f.write('\t')
                    else: f.write('\n')
                
                for row_idx in range(len(curr_table_data)):
                    list_pairs = curr_table_data[row_idx]
                    f.write(self.od[feat_key][row_idx]) # write feat
                    f.write('\t')

                    scores = [''] * nb_classes
                    for class_value_pair in list_pairs:
                        class_idx = class_idx_dict[class_value_pair[0]]
                        assert scores[class_idx] == '' # shouldn't be init.
                        score = str(class_value_pair[1])
                        assert '\t' not in score # would screw up tab-delim
                        scores[class_idx] = score

                    assert all([s != '' for s in scores])
                    f.write('\t'.join(scores))
                    f.write('\n')

# ---------------------------- HELPER FUNCTIONS ------------------------------ #

'''
* Computes orderings of classes from the given 2D array of scores, from most 
  positive to most negative.
* Expects: 
    - score_array = 2D array of scores, where rows (first index) represent class 
      indices, columns (second index) represent feature indices
* Returns:
    - a list of list of pairs; outer list represents the feature, the middle 
      list represent the class, and the inner pair represents (class idx, score)
        e.g., [(0, 100, (3, 90), (2, 80), (1, 70), 
            [(1, 4), (2, 3), (0, 2), (3, 1)]
'''
def compute_orderings(score_array):
    nb_classes = score_array.shape[0]
    nb_features = score_array.shape[1]

    orderings = []
    for feat_idx in range(nb_features):
        # list of (class index, class-specific score) for curr feature 
        class_scores = [(class_idx, score_array[class_idx][feat_idx]) \
            for class_idx in range(nb_classes)]

        ordered_classes = sorted(class_scores, key =lambda x: x[1], reverse=True)

        orderings.append(ordered_classes)

    return orderings

'''
* Computes table of frequency proportion scores from a given 2D frequency table.
  A proportion score is defined as the proportion of samples within a 
  given class (corresponding to row) who exhibit a given feature (corresponding 
  to column). The table is represented as as numpy array with dimensions 
  (nb_classes, nb_features).
* Expects: 
    - class_feat_freq_table = 2D array of frequencies, where rows (first index) 
      represent class indices, columns (second index) represent 
      feature indices
* Returns:
    - table of proportion scores, as described above
'''
def compute_prop_table(class_feat_freq_table):
    nb_features = class_feat_freq_table.shape[1]

    rowsums = np.sum(class_feat_freq_table, axis=1)
    concat_rowsums = np.transpose(np.repeat([rowsums], nb_features, axis=0))
    assert concat_rowsums.shape == class_feat_freq_table.shape

    # important to ensure that numpy array is of type float to prevent
    # integer division
    return class_feat_freq_table.astype(np.float32) / concat_rowsums

'''
* Converts encoded orderings into orderings which have explicit class names.
* Expects: 
    - raw_orderings = list of list of pairs; outer list represents features, 
      middle list represents classes, inner pair represents (class index, score)
        e.g., [(0, 0.9, (3, 0.09), (2, 0.01), (1, 0.0), 
            [(1, 0.5), (2, 0.25), (0, 0.15), (3, 0.1)]
    - idx_class_dict = dictionary mapping class indices to class names
* Returns:
    - a list of list of pairs; outer list represents the feature, the middle 
      list represent the class, and the inner pair represents (class name, score)
        e.g., [('H', 0.9, ('W', 0.09), ('B', 0.01), ('O', 0.0), 
            [('O', 0.5), ('B', 0.25), ('H', 0.15), ('W', 0.1)]
'''
def decode_orderings(raw_orderings, idx_class_dict):
    decoded_orderings = [[(idx_class_dict[pair[0]], pair[1]) for pair in o] \
        for o in raw_orderings]
    return decoded_orderings

# ---------------------------- PUBLIC FUNCTIONS ------------------------------ #

'''
* Creates a OrderingSummary object using various orderings of classes for 
  each feature. 
* Expects: 
    - sums_contribs = 2D array of sums of contribution scores; rows (outer list)
      correspond to features, and columns (inner list) correspond to the class
    - feat_class_freq_table = 2D array of frequencies; rows (outer list)
      correspond to features, and columns (inner list) correspond to the class
    - idx_feat_dict = dictionary mapping feature indices to feature names
    - idx_class_dict = dictionary mapping class indices to class names
    - icd9_descript_dict = dictionary mapping ICD9 codes to their descriptions
    - nb_pairs = number of pairs that were compared
* Returns:
    - a list of list of pairs; outer list represents the feature, the middle 
      list represent the class, and the inner pair represents (class name, score)
        e.g., [('H', 0.9, ('W', 0.09), ('B', 0.01), ('O', 0.0), 
            [('O', 0.5), ('B', 0.25), ('H', 0.15), ('W', 0.1)]
'''
def summarize_orderings(sums_contribs, class_feat_freq_table, idx_feat_dict, 
    idx_class_dict, icd9_descript_dict, nb_pairs):
    class_feat_prop_table = compute_prop_table(class_feat_freq_table)

    list_contrib_orderings = decode_orderings(compute_orderings(sums_contribs),
        idx_class_dict)
    list_freq_orderings = decode_orderings(compute_orderings(class_feat_freq_table), 
        idx_class_dict)
    list_prop_orderings = decode_orderings(compute_orderings(class_feat_prop_table), 
        idx_class_dict)

    list_feat_names = feature_importance.get_list_feat_names(idx_feat_dict, 
        nb_pairs=1) # since we only want 1 repetition
    list_feat_descripts = feature_importance.get_list_feat_descripts(
        list_feat_names, icd9_descript_dict=icd9_descript_dict)

    summary = OrderingSummary(OrderedDict([('feat', list_feat_names),  
        ('descript', list_feat_descripts),  
        ('contrib_ordering', list_contrib_orderings),  
        ('freq_ordering', list_freq_orderings),  
        ('prop_ordering', list_prop_orderings)])) 

    return summary