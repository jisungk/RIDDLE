"""
parameter_search.py

Search for optimal parameters for RIDDLE and various machine learning classifiers.

Requires:   Keras, numpy, scikit-learn, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

import sys; sys.dont_write_bytecode = True
import os
import pickle
import time

FORCE_RUN = False
DATA_DIR = '_data'
CACHE_DIR = '_cache'
SEED = 109971161161043253 % 8085

import numpy as np
np.random.seed(SEED) # for reproducibility, must be before Keras imports!
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
from sklearn.feature_selection import SelectKBest, chi2

from riddle import emr, models, parameter_tuning
from riddle.parameter_tuning import UniformLogSpace, UniformInteger
from pipeline import eprint, pickle_object

# -------------------------- HELPER FUNCTIONS -------------------------------- #

''' 
* Scoring function representing negative loss; used for scikit-learn model
  selection. 
'''
def loss_scorer(estimator, x, y):
    loss = log_loss(y, estimator.predict_proba(x))
    assert loss >= 0
    # minimal loss is best
    # however, we try to maximize the score
    # to account for this we take negative loss
    return -loss

''' 
* Get data for a machine learning pipeline. 
* Expects:
    - data filepath (data_path).
* Returns:
    - data in standard X, y form (X, y)
    - list of permutation indices for shuffling (perm_indices)
    - numbers of features and classes (nb_features, nb_classes)
'''
def get_base_data(data_path, prop_missing):
    icd9_descript_path = '{}/{}'.format(DATA_DIR, 'phewas_codes.txt')

    # load data
    print('Loading data...')
    start = time.time()

    # get common data
    icd9_descript_dict = emr.get_icd9_descript_dict(icd9_descript_path) 
    X, y, idx_feat_dict, idx_class_dict = emr.get_data(path=data_path, 
        icd9_descript_dict=icd9_descript_dict, prop_missing=prop_missing)

    nb_features = len(idx_feat_dict)
    nb_classes = len(idx_class_dict)
    nb_cases = len(X)

    print('Data loaded in {:.5f} s'.format(time.time() - start))
    print()

    # shuffle indices
    perm_indices = np.random.permutation(nb_cases)    
    try: # try validating shuffled indices
        with open(data_path + '_perm_indices.pkl', 'r') as f:
            exp_perm_indices = pickle.load(f)
            assert np.all(perm_indices == exp_perm_indices)
    except:
        eprint('file not found ' + data_path + '_perm_indices.pkl')
        eprint('not doing perm_indices check')

    return X, y, perm_indices, nb_features, nb_classes

''' 
* Vectorizes data for input to the scikit-learn API.
* Expects:
    - data (X, y)
    - number of features (nb_features)
* Returns:
    - preprocessed data in standard X, y form (X, y)
'''
def preproc_for_sklearn(X, y, nb_features):
    try:
        tokenizer = Tokenizer(num_words=nb_features)
    except:
        tokenizer = Tokenizer(num_words=nb_features)
    X = tokenizer.sequences_to_matrix(X, mode='binary')

    return X, y

''' 
* Performs Chi2 feature selection, and gets indices of best features. 
* Expects:
    - data (X, y)
    - number of features (nb_features)
    - number of features to keep (nb_features_to_keep)
* Returns:
    - list of selected feature indices (selected_indices)
'''
def select_feats(X, y, nb_features, nb_features_to_keep=2048):
    X, y = preproc_for_sklearn(X, y, nb_features)

    if nb_features < nb_features_to_keep:
        nb_features_to_keep = nb_features_to_keep / 4

    feature_selector = SelectKBest(chi2, k=nb_features_to_keep).fit(X, y)
    selected_indices = feature_selector.get_support(indices=True) 
    
    return selected_indices

''' 
* Performs parameter search to get best parameters. 
* Expects:
    - parameter tuning data (X_val, y_val)
    - scikit-learn classifier (estimator)
    - parameter search fucntion (search)
    - search space, either a distribution or grid (dist_or_grid)
    - additional arugments for parameter search function (**search_kwargs)
* Returns:
    - dictionary of best parameters
'''
def parameter_search(X_val, y_val, estimator, search, dist_or_grid, **search_kwargs):
    param_search = search(estimator, dist_or_grid, refit=False, **search_kwargs)
    param_search.fit(X_val, y_val)
    return param_search.best_params_


''' 
* Checks of parameter search has been done already. 
* Expects:
    - list of method names (names)
    - string data filename (data_fn)
    - float proportion of data simulated to be missing (prop_missing)
* Returns:
    - boolean, true parameter search has already been done
'''
def already_done(names, data_fn, prop_missing):
    for name in names:
        to_check = '{}/{}_{}_{}_param.pkl'.format(CACHE_DIR, name, 
            data_fn, prop_missing)
        if not os.path.isfile(to_check): return False
    return True

# ---------------------------- PUBLIC FUNCTIONS ------------------------------ #

''' 
* Run parameter search for various machine learning pipelines. 
* Expects:
    - data filepath (data_path)
    - method (method)
    - float proportion of data simulated to be missing (prop_missing)
    - number of partitions for k-fold cross-validation (k)
    - boolean whether to skip nonlinear SVM methods, only relevant if 'svm' is 
      selected as the method (skip_nonlinear_svm)
    - number of searches (nb_searches)
    - maximum number of samples to be used (max_nb_samples)
'''
def run(data_fn, method='lrfc', prop_missing=0.0, k=10, 
    skip_nonlinear_svm=False, max_nb_samples=10000, nb_searches=20):
    if 'dummy' in data_fn or 'debug' in data_fn: nb_searches = 3
    data_path = '{}/{}'.format(DATA_DIR, data_fn)

    if not FORCE_RUN: # check if already did param search, if so, skip 
        did = lambda x: already_done(x, data_fn, prop_missing) # helper
        if method == 'riddle' and did(['riddle']):
            eprint('Already did parameter search for riddle')
            return
        elif method == 'lrfc' and did(['logit', 'rfc']):
            eprint('Already did parameter search for lrfc')
            return
        elif method == 'svm' and did(['linear-svm', 'poly-svm', 'rbf-svm']):
            eprint('Already did parameter search for svm')
            return 

    params = {'riddle': {}, 'logit': {}, 'rfc': {}, 'linear-svm': {},
        'poly-svm': {}, 'rbf-svm': {}}
    X, y, perm_indices, nb_features, nb_classes = get_base_data(data_path, 
        prop_missing)

    for k_idx in range(0, k):
        print('-' * 72)
        print('Partition k = {}'.format(k_idx))
        
        data_partition_dict = emr.get_k_fold_partition(X, y, k_idx=k_idx, k=k, 
            perm_indices=perm_indices)

        X_train = data_partition_dict['X_train']
        y_train = data_partition_dict['y_train']

        X_val = data_partition_dict['X_val']
        y_val = data_partition_dict['y_val']

        # cap number of validation samples
        if max_nb_samples != None and len(X_val)> max_nb_samples:
            X_val = X_val[0:max_nb_samples]
            y_val = y_val[0:max_nb_samples]
        
        if method != 'riddle':
            selected_feat_indices = select_feats(X_train + X_val, y_train + y_val,
                nb_features=nb_features)
            X_val, y_val = preproc_for_sklearn(X_val, y_val, 
                nb_features=nb_features)

            X_val = X_val[:, selected_feat_indices]

        if method == 'riddle':
            start = time.time()
            model_module = models.deep_mlp
            riddle_param_dist = {'learning_rate': UniformLogSpace(10, lo=-6, hi=-1)}
            params['riddle'][k_idx] = parameter_tuning.random_search(model_module, 
                riddle_param_dist, X_val, y_val, nb_features=nb_features, 
                nb_classes=nb_classes, k=3, 
                process_X_data_func_args={'nb_features': nb_features}, 
                process_y_data_func_args={'nb_classes': nb_classes},
                nb_searches=nb_searches)
            print('Best parameters for RIDDLE: {} found in {:.3f} s'
                .format(params['riddle'][k_idx], time.time() - start))

        elif method == 'lrfc':
            # logistic regression
            start = time.time()
            logit_param_dist = {'C': UniformLogSpace()}
            logit_estimator = LogisticRegression(multi_class='multinomial', 
                solver='lbfgs')
            params['logit'][k_idx] = parameter_search(X_val, y_val, 
                estimator=logit_estimator, search=RandomizedSearchCV, 
                dist_or_grid=logit_param_dist, n_iter=nb_searches,
                scoring=loss_scorer)
            print('Best parameters for logistic regression: {} found in {:.3f} s'
                .format(params['logit'][k_idx], time.time() - start))

            # random forest classifier
            start = time.time()
            rfc_param_dist = {'max_features': ['sqrt', 'log2'], \
                'max_depth': UniformLogSpace(base=2, lo=2, hi=9)}
            rfc_estimator = RandomForestClassifier()
            params['rfc'][k_idx] = parameter_search(X_val, y_val, 
                estimator=rfc_estimator, search=RandomizedSearchCV, 
                dist_or_grid=rfc_param_dist, n_iter=nb_searches,
                scoring=loss_scorer)
            print('Best parameters for random forest: {} found in {:.3f} s'
                .format(params['rfc'][k_idx], time.time() - start))

        elif method == 'svm':
            # linear SVM
            start = time.time()
            linear_svm_param_dist = {'C': UniformLogSpace()}
            linear_svm_estimator = SVC(kernel='linear', probability=True)
            params['linear-svm'][k_idx] = parameter_search(X_val, y_val,
                estimator=linear_svm_estimator, search=RandomizedSearchCV, 
                dist_or_grid=linear_svm_param_dist, n_iter=nb_searches, 
                scoring=loss_scorer)
            print('Best parameters for linear SVM: {} found in {:.3f} s'
                .format(params['linear-svm'][k_idx], time.time() - start))

            if skip_nonlinear_svm: continue # skip

            nonlinear_svm_param_dist = {'C': UniformLogSpace(), 
                'gamma': UniformLogSpace(base=10, lo=-5, hi=1)}

            # polynomial SVM
            start = time.time()
            poly_svm_estimator = SVC(kernel='poly', probability=True)
            params['poly-svm'][k_idx] = parameter_search(X_val, y_val,
                estimator=poly_svm_estimator, search=RandomizedSearchCV, 
                dist_or_grid=nonlinear_svm_param_dist, n_iter=nb_searches, 
                scoring=loss_scorer)
            print('Best parameters for polynomial SVM: {} found in {:.3f} s'
                .format(params['poly-svm'][k_idx], time.time() - start))

            # RBF SVM
            start = time.time()
            rbf_svm_estimator = SVC(kernel='rbf', probability=True)
            params['rbf-svm'][k_idx] = parameter_search(X_val, y_val,
                estimator=rbf_svm_estimator, search=RandomizedSearchCV, 
                dist_or_grid=nonlinear_svm_param_dist, n_iter=nb_searches, 
                scoring=loss_scorer)
            print('Best parameters for RBF SVM: {} found in {:.3f} s'
                .format(params['rbf-svm'][k_idx], time.time() - start))

        else: raise ValueError('unknown method: {}'.format(method))

    # save
    for method_name, sub_param_dict in params.items():
        if len(sub_param_dict) > 0:
            pickle_object(sub_param_dict, '{}/{}_{}_{}_param.pkl'.format(
                CACHE_DIR, method_name, data_fn, prop_missing))

    print('Finished parameter search for method: {}'.format(method))

'''
* Runs parameter searches for various machine learning pipelines.
'''
def main(args):
    try: method = args[1].lower()
    except:
        method = 'lrfc'
        eprint('Using default method = \'{}\''.format(method))

    try: data_fn = args[2]
    except: 
        data_fn = 'dummy.txt'
        eprint('Using default data_fn = \'{}\''.format(data_fn))

    try: prop_missing = float(args[3])
    except: 
        prop_missing = 0.0
        eprint('Using default prop_missing = {}'.format(prop_missing))

    # not going to finish in time, so skip nonlinear svm if using full dataset
    skip_nonlinear_svm = 'final-100.txt' in data_fn

    run(data_fn, method=method, prop_missing=prop_missing, 
        skip_nonlinear_svm=skip_nonlinear_svm)

# if run as script, execute main
if __name__ == '__main__':
    import sys

    main(sys.argv)
