"""
parameter_tuning.py

Offers functions to perform parameter tuning through a variety of search
methods. 

Requires:   NumPy, Keras (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

import sys
import time
from itertools import product
import copy

import numpy as np
from scipy.stats import uniform
from keras import backend as K
from .emr import get_k_fold_partition
from .models import train, test

# ------------------------ HELPER FUNCTIONS/CLASSES -------------------------- #

'''
* Class which provides a uniform random number generator. The initialization
  parameter `mass_on_zero` is a floating point number between 0.0 and 1.0 which
  describes what proportion of the time zero should be returned. 
'''
class Uniform(object):
    def __init__(self, lo=0, hi=1, mass_on_zero=0.0):
        self.lo = lo
        self.scale = hi - lo
        self.mass_on_zero = mass_on_zero

    def rvs(self, random_state=None):
        if random_state is None:
            gen = uniform(loc=self.lo, scale=self.scale).rvs()
        else:
            gen = uniform(loc=self.lo, scale=self.scale).rvs(random_state=random_state)

        if self.mass_on_zero > 0.0 and np.random.uniform() < self.mass_on_zero:
            return 0.0

        return gen

'''
* Class which provides a uniform random integer generator. Inherits initialization
  parameters from the Uniform class. 
'''
class UniformInteger(Uniform):
    def rvs(self, random_state=None):
        return int(super(UniformInteger, self).rvs(random_state))

'''
* Class which provides a uniform-in-log-space random number generator. The
  initialization parameter `mass_on_zero` is a floating point number between 0.0
  and 1.0 which describes what proportion of the time zero should be returned. 
'''
class UniformLogSpace(object):
    def __init__(self, base=10, lo=-3, hi=3, mass_on_zero=0.0):
        self.base = base
        self.lo = lo
        self.scale = hi - lo
        self.mass_on_zero = mass_on_zero

    def rvs(self, random_state=None):
        if random_state is None:
            exp = uniform(loc=self.lo, scale=self.scale).rvs()
        else:
            exp = uniform(loc=self.lo, scale=self.scale).rvs(random_state=random_state)

        if self.mass_on_zero > 0.0 and np.random.uniform() < self.mass_on_zero:
            return 0.0

        return self.base ** exp

'''
* Class which provides a uniform-in-log-space random integer generator. Inherits
  initialization parameters from the UniformLogSpace class. 
'''
class UniformIntegerLogSpace(UniformLogSpace):
    def rvs(self, random_state=None):
        return int(super(UniformIntegerLogSpace, self).rvs(random_state))

''' 
* Creates all possible combinations from a grid of parameters.
* Expects:
    - param_grid = grid of parameters
* Returns:
    - list of dictionaries representing setting combinations
'''
def param_combinations(param_grid):
    param_idx_dict = {}
    list_param_grid = []
    for idx, (key, values) in enumerate(param_grid.items()):
        param_idx_dict[idx] = key
        list_param_grid.append(values)
    
    nb_param = len(param_grid.keys())
    param_combos = []
    for param_values in list(product(*list_param_grid)):
        assert len(param_values) == nb_param

        curr_param_set = {}
        for idx, v in enumerate(param_values):
            curr_param_set[param_idx_dict[idx]] = v

        param_combos.append(curr_param_set)

    return param_combos

# -------------------------------- FUNCTIONS --------------------------------- #

''' 
* Finds optimal parameters using exhaustive grid search.
* Expects:
    - model_module = model module
    - param_grid = dictionary grid of parameters
    - X = feature data
    - y = class data 
    - nb_features = number of features
    - nb_classes = number of classes
    - k = number of partitions for k-fold cross-validation
    - process_X_data_func = function to process feature data
    - process_y_data_func = function to process class data
    - process_X_data_func_args = additional arguments for process_X_data_func(); 
      note: X is already passed as the first argument
    - process_y_data_func_args = additional arguments for process_y_data_func(); 
      note: y is already passed as the first argument
    - max_nb_samples = maximum number of samples to be used
* Returns:
    - dictionary of best parameters
'''
def grid_search(model_module, param_grid, X, y, nb_features, nb_classes, 
    k=3, process_X_data_func_args={}, process_y_data_func_args={}, 
    max_nb_samples=10000):
    model_init_function = model_module.create_base_model
    process_X_data_func = model_module.process_X_data
    process_y_data_func = model_module.process_y_data

    if len(X) > max_nb_samples:
        X = X[:max_nb_samples]
        y = y[:max_nb_samples]

    nb_cases = len(X)
    perm_indices = np.random.permutation(nb_cases)

    param_combos = param_combinations(param_grid) # pass by value

    best_param_set = None
    best_loss = sys.float_info.max
    worst_loss = -1
    for param_set_idx, param_set in enumerate(param_combos):
        start = time.time()
        losses = []
        accs = []
        for k_idx in range(k):
            data_partition_dict = get_k_fold_partition(X, y, k_idx=k_idx, k=k, 
                perm_indices=perm_indices)

            X_train = data_partition_dict['X_train']
            y_train = data_partition_dict['y_train']
            X_val = data_partition_dict['X_val']
            y_val = data_partition_dict['y_val']
            X_test = data_partition_dict['X_test']
            y_test = data_partition_dict['y_test']

            model = model_init_function(nb_features=nb_features, 
                nb_classes=nb_classes, **param_set)

            model = train(model, X_train, y_train, X_val, y_val, 
                process_X_data_func, process_y_data_func, 
                nb_features=nb_features, nb_classes=nb_classes, 
                process_X_data_func_args=process_X_data_func_args,
                process_y_data_func_args=process_y_data_func_args,
                verbose=False)

            (loss, acc), y_test_proba = test(model, X_test, 
                y_test, process_X_data_func, process_y_data_func, 
                nb_features=nb_features, nb_classes=nb_classes, 
                process_X_data_func_args=process_X_data_func_args,
                process_y_data_func_args=process_y_data_func_args,
                verbose=False)

            K.clear_session()

            losses.append(loss)
            accs.append(acc)

        avg_loss = np.mean(losses)
        avg_acc = np.mean(accs)

        if avg_loss < best_loss:
            best_param_set = param_set
            best_loss = avg_loss

        if avg_loss > worst_loss:
            worst_loss = avg_loss

    print('During parameter tuning, best loss: {:.4f} / Worst loss: {:.4f}'
        .format(best_loss, worst_loss))

    return best_param_set


''' 
* Finds optimal parameters using random grid search.
* Expects:
    - model_module = model module
    - dictionary of parameter distributions (param_dist)
    - X = feature data
    - y = class data 
    - nb_features = number of features
    - nb_classes = number of classes
    - k = number of partitions for k-fold cross-validation
    - process_X_data_func = function to process feature data
    - process_y_data_func = function to process class data
    - process_X_data_func_args = additional arguments for process_X_data_func(); 
      note: X is already passed as the first argument
    - process_y_data_func_args = additional arguments for process_y_data_func(); 
      note: y is already passed as the first argument
    - max_nb_samples = maximum number of samples to be used
    - nb_searches = number of searches
* Returns:
    - dictionary of best parameters
'''
def random_search(model_module, param_dist, X, y, nb_features, 
    nb_classes, k=3, process_X_data_func_args={}, process_y_data_func_args={}, 
    max_nb_samples=10000, nb_searches=20):
    model_init_function = model_module.create_base_model
    process_X_data_func = model_module.process_X_data
    process_y_data_func = model_module.process_y_data

    if len(X) > max_nb_samples:
        X = X[:max_nb_samples]
        y = y[:max_nb_samples]

    nb_cases = len(X)
    perm_indices = np.random.permutation(nb_cases)

    best_param_set = None
    best_loss = sys.float_info.max
    worst_loss = -1
    for search_idx in range(nb_searches): 
        start = time.time()

        param_set = {}
        for param, values in param_dist.items():
            if hasattr(values, '__getitem__'): # list
                param_set[param] = values[np.random.randint(0, len(values))]
            elif hasattr(values, 'rvs'): # distributino
                param_set[param] = values.rvs()
            else: # single item
                param_set[param] = values

        losses = []
        accs = []
        for k_idx in range(k):
            data_partition_dict = get_k_fold_partition(X, y, k_idx=k_idx, k=k, 
                perm_indices=perm_indices)

            X_train = data_partition_dict['X_train']
            y_train = data_partition_dict['y_train']
            X_val = data_partition_dict['X_val']
            y_val = data_partition_dict['y_val']
            X_test = data_partition_dict['X_test']
            y_test = data_partition_dict['y_test']

            model = model_init_function(nb_features=nb_features, 
                nb_classes=nb_classes, **param_set)

            model = train(model, X_train, y_train, X_val, y_val, 
                process_X_data_func, process_y_data_func, 
                nb_features=nb_features, nb_classes=nb_classes, 
                process_X_data_func_args=process_X_data_func_args,
                process_y_data_func_args=process_y_data_func_args,
                verbose=False)

            (loss, acc), y_test_proba = test(model, X_test, 
                y_test, process_X_data_func, process_y_data_func, 
                nb_features=nb_features, nb_classes=nb_classes, 
                process_X_data_func_args=process_X_data_func_args,
                process_y_data_func_args=process_y_data_func_args,
                verbose=False)

            losses.append(loss)
            accs.append(acc)

            K.clear_session()

        avg_loss = np.mean(losses)
        avg_acc = np.mean(accs)

        if avg_loss < best_loss:
            best_param_set = param_set
            best_loss = avg_loss

        if avg_loss > worst_loss:
            worst_loss = avg_loss

    print('During parameter tuning, best loss: {:.4f} / Worst loss: {:.4f}'
        .format(best_loss, worst_loss))

    return best_param_set