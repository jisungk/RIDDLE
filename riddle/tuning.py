"""tuning.py

Offers functions to perform parameter tuning through a variety of search
methods.

Requires:   NumPy, Keras (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

from __future__ import print_function

from itertools import product

import numpy as np
from keras import backend as K
from scipy.stats import uniform
from sklearn.metrics import log_loss

from .emr import get_k_fold_partition


class Uniform(object):
    """Uniform random floating number generator."""

    def __init__(self, lo=0, hi=1, mass_on_zero=0.0):
        """Initialize random uniform floating number generator.

        Arguments:
            lo: float
                lowest number in range
            hi: float
                highest number in range
            mass_on_zero: float
                probability that zero be returned
        """
        self.lo = lo
        self.scale = hi - lo
        self.mass_on_zero = mass_on_zero

    def rvs(self, random_state=None):
        """Draw a value from this random variable."""
        if self.mass_on_zero > 0.0 and np.random.uniform() < self.mass_on_zero:
            return 0.0
        return uniform(loc=self.lo, scale=self.scale).rvs(
            random_state=random_state)


class UniformInteger(Uniform):
    """Uniform random integer generator."""

    def rvs(self, random_state=None):
        return int(super(UniformInteger, self).rvs(random_state))


class UniformLogSpace(object):
    """Random floating number generator which is uniform in logspace."""

    def __init__(self, base=10, lo=-3, hi=3, mass_on_zero=0.0):
        """Initialize random floating number generator.

        Arguments:
        - base: int
            base of number
        - lo: float
            lowest exponent in range
        - hi: float
            highest exponent in range
        - mass_on_zero: float
            probability that zero be returned
        """
        self.base = base
        self.lo = lo
        self.scale = hi - lo
        self.mass_on_zero = mass_on_zero

    def rvs(self, random_state=None):
        if random_state is None:
            exp = uniform(loc=self.lo, scale=self.scale).rvs()
        else:
            exp = uniform(loc=self.lo, scale=self.scale).rvs(
                random_state=random_state)

        if self.mass_on_zero > 0.0 and np.random.uniform() < self.mass_on_zero:
            return 0.0

        return self.base ** exp


class UniformIntegerLogSpace(UniformLogSpace):
    """Random integer generator which is uniform in logspace."""

    def rvs(self, random_state=None):
        return int(super(UniformIntegerLogSpace, self).rvs(random_state))


def grid_search(model_class, init_args, param_grid, x_unvec, y,
                num_class, k=3, max_num_sample=10000):
    """Finds optimal parameters via an exhuastive grid search.

    Arguments:
        model_class: class
            class of RIDDLE model
        init_args: {string: ?}
            dictionary mapping initialization argument names to their values
        param_grid: {string: ?}
            dictionary mapping parameter names to values or lists of possible
            values
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        y: [int]
            list of class labels as integer indices
        k: int
            number of partitiosn for k-fold cross-validation
        max_num_sample: int
            maximum number of samples to use

    Returns:
        best_param: {string: ?}
            dictionary mapping parameter names to the best values found
    """
    param_list = _param_combinations(param_grid)

    best_param_set, best_loss, worst_loss = _search(
        model_class, init_args, param_list, x_unvec, y, num_class=num_class,
        k=k, max_num_sample=max_num_sample)

    print('During parameter tuning, best loss: {:.4f} / Worst loss: {:.4f}'
          .format(best_loss, worst_loss))

    return best_param_set


def random_search(model_class, init_args, param_dist, x_unvec, y,
                  num_class, k=3, max_num_sample=10000, num_search=20):
    """Finds optimal parameters via a random grid search.

    Arguments:
        model_class: class
            class of RIDDLE model
        init_args: {string: ?}
            dictionary mapping initialization argument names to their values
        param_dist: {string: ?}
            dictionary mapping parameter names to values, random variables
            which generate values or lists of possible values
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        y: [int]
            list of class labels as integer indices
        k: int
            number of partitiosn for k-fold cross-validation
        max_num_sample: int
            maximum number of samples to use
        num_search: int
            number of searches (parameter configurations) to try

    Returns:
        best_param: {string: ?}
            dictionary mapping parameter names to the best values found
    """
    param_list = []
    for _ in range(num_search):
        param = {}
        for param_name, values in param_dist.items():
            if hasattr(values, '__getitem__'):  # list
                param[param_name] = values[np.random.randint(0, len(values))]
            elif hasattr(values, 'rvs'):  # distributino
                param[param_name] = values.rvs()
            else:  # single item
                param[param_name] = values
        param_list.append(param)

    best_param_set, best_loss, worst_loss = _search(
        model_class, init_args, param_list, x_unvec, y, num_class=num_class,
        k=k, max_num_sample=max_num_sample)

    print('During parameter tuning, best loss: {:.4f} / Worst loss: {:.4f}'
          .format(best_loss, worst_loss))

    return best_param_set


def _param_combinations(param_grid):
    """Generate all possible combinations from a grid of parameters.

    Arguments:
        param_grid: {string: [?]}
            dictionary mapping parameters to a list of their values

    Returns:
        param_list: [{string: ?}]
            list of possible parameters represented as dicts
    """
    param_idx_dict = {}
    list_param_grid = []
    for idx, (key, values) in enumerate(param_grid.items()):
        param_idx_dict[idx] = key
        list_param_grid.append(values)

    num_param = len(param_grid.keys())
    param_combos = []
    for param_values in list(product(*list_param_grid)):
        assert len(param_values) == num_param

        curr_param_set = {}
        for idx, v in enumerate(param_values):
            curr_param_set[param_idx_dict[idx]] = v

        param_combos.append(curr_param_set)

    return param_combos


def _search(model_class, init_args, param_list, x_unvec, y, num_class, k=3,
            max_num_sample=10000):
    """Finds optimal parameters by trying the given configurations.

    Arguments:
        model_class: class
            class of RIDDLE model
        init_args: {string: ?}
            dictionary mapping initialization argument names to their values
        param_list: [{string: ?}]
            list of parameter configurations represent as dictionaries mapping
            parameter names to values
        x_unvec: [[int]]
            feature indices that have not been vectorized; each inner list
            collects the indices of features that are present (binary on)
            for a sample
        y: [int]
            list of class labels as integer indices
        num_class: int
            number of classes
        k: int
            number of partitiosn for k-fold cross-validation
        max_num_sample: int
            maximum number of samples to use

    Returns:
        best_param: {string: ?}
            dictionary mapping parameter names to the best values found
    """

    if len(x_unvec) > max_num_sample:
        x_unvec = x_unvec[:max_num_sample]
        y = y[:max_num_sample]

    num_sample = len(x_unvec)
    perm_indices = np.random.permutation(num_sample)

    best_param_set = None
    best_loss = float('inf')
    worst_loss = -1
    for param_set in param_list:
        y_test_probas_all = np.empty([0, num_class])
        y_test_all = np.empty([0, ])
        for k_idx in range(k):
            x_train_unvec, y_train, x_val_unvec, y_val, x_test_unvec, y_test = (
                get_k_fold_partition(x_unvec, y, k_idx=k_idx, k=k,
                                     perm_indices=perm_indices))

            full_init_args = dict(init_args, **param_set)
            model = model_class(**full_init_args)
            model.train(x_train_unvec, y_train, x_val_unvec, y_val,
                        verbose=False)
            y_test_probas = model.predict_proba(x_test_unvec, verbose=False)

            y_test_probas_all = np.append(
                y_test_probas_all, y_test_probas, axis=0)
            y_test_all = np.append(y_test_all, y_test, axis=0)

            K.clear_session()

        loss = log_loss(y_test_all, y_test_probas_all, labels=range(num_class))
        if loss < best_loss:
            best_param_set = param_set
            best_loss = loss

        if loss > worst_loss:
            worst_loss = loss

    return best_param_set, best_loss, worst_loss
