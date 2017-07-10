"""
pipeline.py

A centralized pipeline for data acquisition, processing, model training, and 
model testing using RIDDLE. 

Requires:   NumPy, scikit-learn, RIDDLE (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

import sys; sys.dont_write_bytecode = True
import os
import time
import pickle

DATA_DIR = '_data'
SEED = 109971161161043253 % 8085

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ---------------------------- HELPER FUNCTIONS ------------------------------ #

'''
* Prints to standard error.
'''
def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

'''
* Pickles an object to file.
* Expects:
    - obj = object
    - fn = filename
'''
def pickle_object(obj, fn):
    with open(fn, 'w') as f:
        pickle.dump(obj, f)

''' 
* Perform a detailed evaluation of the model, creating a confusion matrix, 
  a sklearn classification report, and ROC curves. Prints and/or saves relevant
  information.
* Expects:
    - y_test = list of true targets
    - y_test_proba = probability vectors
    - nb_classes = number of classes
    - path = string path where output plots should be saved
'''
def evaluate(y_test, y_test_proba, nb_classes, path):
    from riddle import roc # here so np can be seeded before run_pipeline() call

    y_pred = [np.argmax(p) for p in y_test_proba]

    print('Confusion matrix:')
    print(confusion_matrix(y_test, y_pred))
    print()

    print('Classification report:')
    print(classification_report(y_test, y_pred, digits=3))

    print('ROC AUC values:')
    roc_auc, fpr, tpr = roc.compute_roc(y_test, y_test_proba, 
        nb_classes=nb_classes)
    roc.save_plots(roc_auc, fpr, tpr, nb_classes=nb_classes, path=path)

    for l, r in roc_auc.items():
        print('  {}: {:.5f}'.format(l, r))
    print()

# ---------------------------- PUBLIC FUNCTIONS ------------------------------ #

''' 
* Run a full deep learning pipeline (acquire data, process, train, test, 
  evaluate). 
* Expects:
    - model_module = module which contains functions to:
        + initalize a compiled Keras Sequential model 
          (model_module.create_base_model)
        + process feature data to appropriate form (model_module.process_X_data)
        + process class data to appropriate form (model_module.process_y_data)
    - best_model_param = dictionary of best parameters for model_module
    - data_partition_dict = dictionary of the train, validation, and test data
      which should have these keys (with the appropriate value):
        + X_train
        + y_train
        + X_val
        + y_val
        + X_test
        + y_test
    - nb_features = number of features
    - nb_classes = number of classes
    - interpret_model = boolean whether to compute feature importance scores
    - out_directory = string path where output should be saved
    - max_nb_epoch = maximum number of epochs
* Return
    - tuple of metrics (loss, accuracy, runtime) 
    - tuple of sums of differences and sums of deepLIFT contrib scores; 
      these are used for t-tests in feature interpretation 
      (sums_D, sums_D2, sums_contrib)
    - pairs of compared classes (pairs)
'''
def run_pipeline(model_module, best_model_param, data_partition_dict, 
    nb_features, nb_classes, interpret_model, out_directory, max_nb_epoch=100):
    # here so np can be seeded before keras imports
    from scipy.stats import uniform, randint
    from keras import backend as K
    from riddle import models, feature_importance

    # ----------------------------- EXTRACT DATA ----------------------------- #
    X_train = data_partition_dict['X_train']
    y_train = data_partition_dict['y_train']
    X_val = data_partition_dict['X_val']
    y_val = data_partition_dict['y_val']
    X_test = data_partition_dict['X_test']
    y_test = data_partition_dict['y_test']

    print('{} train samples / {} val samples / {} test samples'
        .format(len(X_train), len(X_val), len(X_test)))
    print('{} features / {} classes'.format(nb_features, nb_classes))
    print()

    # -------------------------------- SETUP --------------------------------- #
    process_X_data_func = model_module.process_X_data
    process_y_data_func = model_module.process_y_data
    
    process_X_data_func_args = {'nb_features': nb_features}
    process_y_data_func_args = {'nb_classes': nb_classes}

    model = model_module.create_base_model(nb_features=nb_features, 
        nb_classes=nb_classes, **best_model_param)

    # ----------------------------- TRAIN MODEL ------------------------------ #
    start = time.time()

    if 'debug' in out_directory or 'dummy' in out_directory: max_nb_epoch = 3

    model = models.train(model, X_train, y_train, X_val, y_val, 
        process_X_data_func, process_y_data_func, nb_features=nb_features, 
        nb_classes=nb_classes, process_X_data_func_args=process_X_data_func_args,
        process_y_data_func_args=process_y_data_func_args, 
        max_nb_epoch=max_nb_epoch)

    # -------------------------- TEST/EVALUATE MODEL ------------------------- #
    (loss, acc), y_test_proba = models.test(model, X_test, 
        y_test, process_X_data_func, process_y_data_func, nb_features=nb_features, 
        nb_classes=nb_classes, process_X_data_func_args=process_X_data_func_args,
        process_y_data_func_args=process_y_data_func_args)

    runtime = time.time() - start
    print('Completed training and testing in {:.4f} seconds'.format(runtime))  
    print('-' * 72)
    print()

    # save results
    test_results_path = out_directory + '/test_results.txt'
    models.save_test_results(y_test_proba, y_test, path=test_results_path)
    
    # evaluate model performance
    roc_graph_path = out_directory + '/roc.png'
    evaluate(y_test, y_test_proba, nb_classes=nb_classes, path=roc_graph_path)

    # ------------------------ FEATURE IMPORTANCE PREP ----------------------- #
    start = time.time()

    if interpret_model:
        sums_D, sums_D2, sums_contribs, pairs = \
            feature_importance.get_diff_sums(model, X_test, process_X_data_func, 
                nb_features=nb_features, nb_classes=nb_classes, 
                process_X_data_func_args=process_X_data_func_args)

        pickle_object(sums_D, out_directory + '/sums_D.pkl')
        pickle_object(sums_D2, out_directory + '/sums_D2.pkl')
        pickle_object(sums_contribs, out_directory + '/sums_contribs.pkl')

        print('Computed deepLIFT scores and pre-analysis in {:.4f} seconds'
            .format(time.time() - start))
        print('-' * 72)
        print()
    else:
        sums_D, sums_D2, sums_contribs, pairs = None, None, None, None

    # ------------------------------ SAVE MODEL ------------------------------ #
    model_path = out_directory + '/model.h5'
    models.save_model(model, path=model_path)

    K.clear_session()

    return (loss, acc, runtime), (sums_D, sums_D2, sums_contribs, pairs)
    
# not implemented yet
def main():
    # seed here to avoid re-seeding when run_pipeline() is called by other files
    np.random.seed(SEED) # for reproducibility, must be before Keras imports!

    pass # TODO write this

# if run as script, execute main
if __name__ == '__main__':
    import sys
    main(sys.argv)