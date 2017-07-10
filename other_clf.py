"""
other_clf.py

Run various machine learning classifier pipelines.

Requires:   Keras, NumPy, scikit-learn, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

import sys; sys.dont_write_bytecode = True
import os
import pickle
import time

DATA_DIR = '_data'
CACHE_DIR = '_cache'
SEED = 109971161161043253 % 8085

import numpy as np
np.random.seed(SEED) # for reproducibility, must be before Keras imports!
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
from sklearn.externals import joblib

from riddle import emr
from riddle.parameter_tuning import UniformLogSpace, UniformInteger
from riddle.models import save_test_results

from parameter_search import loss_scorer, get_base_data, preproc_for_sklearn, select_feats
from pipeline import eprint
from kfold_pipeline import print_metrics

# ---------------------------- PUBLIC FUNCTIONS ------------------------------ #

''' 
* Run parameter search for various machine learning pipelines. 
* Expects:
    - data_path = data filepath
    - method = method
    - which_half = string setting for whether to do first half, last half, or 
      the complete set of experiments; one of ('first', 'last', 'both')
    - prop_missing = float proportion of data simulated to be missing
    - k = number of partitions for k-fold cross-validation
    - skip_nonlinear_svm = boolean whether to skip nonlinear SVM methods, 
      only relevant if 'svm' is selected as the method
    - nb_searches = number of searches
'''
def run(data_fn, method='lrfc', which_half='both', prop_missing=0.0, k=10, 
    skip_nonlinear_svm=False, nb_searches=20):
    data_path = '{}/{}'.format(DATA_DIR, data_fn)

    def get_results_dir(method, k_idx):
        base_folder = 'out/more/{}_{}_{}'.format(method, data_fn, prop_missing)
        folder = '{}/{}_idx_partition'.format(base_folder, k_idx)

        if not os.path.exists('out'): os.makedirs('out')
        if not os.path.exists('out/more'): os.makedirs('out/models')
        if not os.path.exists(base_folder): os.makedirs(base_folder)
        if not os.path.exists(folder): os.makedirs(folder)

        return folder

    try: # load saved parameters
        get_param_fn = lambda x: '{}/{}_{}_{}_param.pkl'.format(CACHE_DIR, 
            x, data_fn, prop_missing)

        if method == 'lrfc':
            with open(get_param_fn('logit'), 'r') as f:
                logit_params = pickle.load(f)
            with open(get_param_fn('rfc'), 'r') as f:
                rfc_params = pickle.load(f)
        elif method == 'svm':
            with open(get_param_fn('linear-svm'), 'r') as f:
                linear_svm_params = pickle.load(f)
            if not skip_nonlinear_svm:
                with open(get_param_fn('poly-svm'), 'r') as f:
                    poly_svm_params = pickle.load(f)
                with open(get_param_fn('rbf-svm'), 'r') as f:
                    rbf_svm_params = pickle.load(f)
        else: raise ValueError('unknown method: {}'.format(method))
    except:
        eprint('Need to do parameter search!')
        eprint('Please run `parameter_search.py` with the relevant' + 
               'command line arguments')
        raise

    X, y, perm_indices, nb_features, nb_classes = get_base_data(data_path, 
        prop_missing)

    losses = {'logit':[], 'rfc':[], 'linear-svm':[], 'poly-svm':[], 'rbf-svm':[]}
    accs = {'logit':[], 'rfc':[], 'linear-svm':[], 'poly-svm':[], 'rbf-svm':[]}
    runtimes = {'logit':[], 'rfc':[], 'linear-svm':[], 'poly-svm':[], 'rbf-svm':[]}

    if which_half == 'first': loop_seq = range(0, k / 2)
    elif which_half == 'last': loop_seq = range(k / 2, k)
    elif which_half == 'both': loop_seq = range(0, k)
    else: raise ValueError('`which_half` must be \'first\', \'last\' or \'both\'')

    for k_idx in loop_seq:
        print('-' * 72)
        print('Partition k = {}'.format(k_idx))

        data_partition_dict = emr.get_k_fold_partition(X, y, k_idx=k_idx, k=k, 
        perm_indices=perm_indices)
        X_train = data_partition_dict['X_train']
        y_train = data_partition_dict['y_train']
        X_val   = data_partition_dict['X_val']
        y_val   = data_partition_dict['y_val']
        X_test  = data_partition_dict['X_test']
        y_test  = data_partition_dict['y_test']

        selected_feat_indices = select_feats(X_train + X_val, y_train + y_val,
            nb_features=nb_features)

        X_train, y_train = preproc_for_sklearn(X_train, y_train, nb_features)
        X_test, y_test = preproc_for_sklearn(X_test, y_test, nb_features)

        old_nb_features = len(X_train[0])
        X_train = X_train[:, selected_feat_indices]
        X_test = X_test[:, selected_feat_indices]

        nb_features = len(X_train[0]) # extraneous but for future utility
        print('Reduced features from {} to {}'.format(old_nb_features, nb_features))

        if method == 'lrfc':
            from sklearn.linear_model import LogisticRegression
            from sklearn.ensemble import RandomForestClassifier
            
            # logistic regression
            start = time.time()
            logit = LogisticRegression(multi_class='multinomial', solver='lbfgs', 
                **logit_params[k_idx])
            logit.fit(X_train, y_train)
            logit_acc = accuracy_score(y_test, logit.predict(X_test))
            logit_y_test_proba = logit.predict_proba(X_test)
            logit_loss = log_loss(y_test, logit_y_test_proba)
            logit_time = time.time() - start
            print('Logistic regression / loss: {:.3f} / accuracy: {:.3f} / time: {:.3f} s'
                .format(logit_loss, logit_acc, logit_time))

            # random forest classifier
            start = time.time()
            rfc = RandomForestClassifier(**rfc_params[k_idx])
            rfc.fit(X_train, y_train)
            rfc_acc = accuracy_score(y_test, rfc.predict(X_test))
            rfc_y_test_proba = rfc.predict_proba(X_test)
            rfc_loss = log_loss(y_test, rfc_y_test_proba)
            rfc_time = time.time() - start
            print('Random forest / loss: {:.3f} / accuracy: {:.3f} / time: {:.3f} s'
                .format(rfc_loss, rfc_acc, rfc_time))
            
            save_test_results(logit_y_test_proba, y_test, 
                '{}/test_results.txt'.format(get_results_dir('logit', k_idx)))
            save_test_results(rfc_y_test_proba, y_test, 
                '{}/test_results.txt'.format(get_results_dir('rfc', k_idx)))
            # joblib.dump(logit, get_results_dir('logit', k_idx) + '/clf.pkl')
            # joblib.dump(rfc, get_results_dir('rfc', k_idx) + '/clf.pkl')

            losses['logit'].append(logit_loss)
            accs['logit'].append(logit_acc)
            runtimes['logit'].append(logit_time)

            losses['rfc'].append(rfc_loss)
            accs['rfc'].append(rfc_acc)
            runtimes['rfc'].append(rfc_time)

        elif method == 'svm':
            from sklearn.svm import SVC

            # linear SVM
            start = time.time()
            linear_svm = SVC(kernel='linear', probability=True, 
                **linear_svm_params[k_idx])
            linear_svm.fit(X_train, y_train)
            linear_svm_acc = accuracy_score(y_test, linear_svm.predict(X_test))
            linear_svm_y_test_proba = linear_svm.predict_proba(X_test)
            linear_svm_loss = log_loss(y_test, linear_svm_y_test_proba)
            linear_svm_time = time.time() - start
            print('Linear SVM / accuracy: {:.3f} / loss: {:.3f} / time: {:.3f} s'
                .format(linear_svm_acc, linear_svm_loss, linear_svm_time))

            save_test_results(linear_svm_y_test_proba, y_test, 
                '{}/test_results.txt'.format(get_results_dir('linear-svm', k_idx)))
            # joblib.dump(linear_svm, get_results_dir('linear-svm', k_idx) + '/clf.pkl')

            losses['linear-svm'].append(linear_svm_loss)
            accs['linear-svm'].append(linear_svm_acc)
            runtimes['linear-svm'].append(linear_svm_time)

            if skip_nonlinear_svm: continue # skip

            # polynomial SVM
            start = time.time()
            poly_svm = SVC(kernel='poly', probability=True,
                **poly_svm_params[k_idx])
            poly_svm.fit(X_train, y_train)
            poly_svm_acc = accuracy_score(y_test, poly_svm.predict(X_test))
            poly_svm_y_test_proba = poly_svm.predict_proba(X_test)
            poly_svm_loss = log_loss(y_test, poly_svm_y_test_proba)
            poly_svm_time = time.time() - start
            print('Polynomial SVM / accuracy: {:.3f} / loss: {:.3f} / time: {:.3f} s'
                .format(poly_svm_acc, poly_svm_loss, poly_svm_time))

            # RBF SVM
            start = time.time()
            rbf_svm = SVC(kernel='rbf', probability=True, 
                **rbf_svm_params[k_idx])
            rbf_svm.fit(X_train, y_train)
            rbf_svm_acc = accuracy_score(y_test, rbf_svm.predict(X_test))
            rbf_svm_y_test_proba = rbf_svm.predict_proba(X_test)
            rbf_svm_loss = log_loss(y_test, rbf_svm_y_test_proba)
            rbf_svm_time = time.time() - start
            print('RBF SVM / accuracy: {:.3f} / loss: {:.3f} / time: {:.3f} s'
                .format(rbf_svm_acc, rbf_svm_loss, rbf_svm_time))

            save_test_results(poly_svm_y_test_proba, y_test, 
                '{}/test_results.txt'.format(get_results_dir('poly-svm', k_idx)))
            save_test_results(rbf_svm_y_test_proba, y_test, 
                '{}/test_results.txt'.format(get_results_dir('rbf-svm', k_idx)))
            # joblib.dump(poly_svm, get_results_dir('poly-svm', k_idx) + '/clf.pkl')
            # joblib.dump(rbf_svm, get_results_dir('rbf-svm', k_idx) + '/clf.pkl')

            losses['poly-svm'].append(poly_svm_loss)
            accs['poly-svm'].append(poly_svm_acc)
            runtimes['poly-svm'].append(poly_svm_time)

            losses['rbf-svm'].append(rbf_svm_loss)
            accs['rbf-svm'].append(rbf_svm_acc)
            runtimes['rbf-svm'].append(rbf_svm_time)

        else: raise ValueError('unknown method: {}'.format(method))

    print()
    print('#' * 72)
    if method == 'lrfc':
        print_metrics(losses['logit'], accs['logit'], runtimes['logit'],
            'Logistic regression')
        print_metrics(losses['rfc'], accs['rfc'], runtimes['rfc'], 
            'Random forest')
    elif method == 'svm':
        print_metrics(losses['linear-svm'], accs['linear-svm'], 
            runtimes['linear-svm'], 'Linear SVM')
        if not skip_nonlinear_svm:
            print_metrics(losses['poly-svm'], accs['poly-svm'], 
                runtimes['poly-svm'], 'Polynomial SVM')
            print_metrics(losses['rbf-svm'], accs['rbf-svm'], 
                runtimes['rbf-svm'], 'RBF SVM')
    else: raise ValueError('unknown method: {}'.format(method))
    print('#' * 72)

'''
* Runs parameter searches for various machine learning pipelines.
> Command line arguments:
    + method = one of ('lrfc', 'svm')
    + data_fn = string data file name
    + which_half = string setting for whether to do first half, last half, or 
      the complete set of experiments; one of ('first', 'last', 'both')
    + prop_missing = float proportion of data to randomly simulate as missing
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

    try: which_half = args[3].lower()
    except: 
        which_half = 'both'
        eprint('Using default which_half = {}'.format(which_half))

    try: prop_missing = float(args[4])
    except: 
        prop_missing = 0.0
        eprint('Using default prop_missing = {}'.format(prop_missing))

    # not going to finish in time, so skip nonlinear svm if using full dataset
    skip_nonlinear_svm = 'final-100.txt' in data_fn
    if skip_nonlinear_svm:
        eprint('Skipping SVMs with non-linear kernels')
        
    run(data_fn, method=method, which_half=which_half,
        prop_missing=prop_missing, skip_nonlinear_svm=skip_nonlinear_svm)

# if run as script, execute main
if __name__ == '__main__':
    import sys

    main(sys.argv)
