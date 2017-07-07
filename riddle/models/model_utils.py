"""
model_utils.py

Provides useful, common functions for training and testing Keras models.

Requires:   NumPy, Keras (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

import sys
from math import ceil
import time
from itertools import izip

import numpy as np
from keras.models import Sequential, load_model as keras_load_model

# -------------------------------- FUNCTIONS --------------------------------- #

''' 
* A generator which yields n-sized chunks a list.
* Expects:
    - l = list
    - n = number of samples per chunk/batch
'''
def chunks(l, n):
    assert n > 0
    
    for i in range(0, len(l), n):
        chunk = l[i:i+n]
        yield chunk

'''
* Trains a Keras model in a batch-wise fashion. It is too expensive to 
  vectorize the entire train and validation set, so the data is vectorized in 
  a streaming fashion as needed (e.g., in sync with training) using a generator.
* Expects:
    - model = Sequential Keras model
    - X_train, X_val = train and validation sets of feature data
    - y_train, y_val = train and validation sets of class data
    - process_X_data_func = function to process feature data
    - process_y_data_func = function to process class data
    - nb_features = number of features
    - nb_classes = number of classes
    - process_X_data_func_args = additional arguments for process_X_data_func(); 
      note: X is already passed as the first argument
    - process_y_data_func_args = additional arguments for process_y_data_func(); 
      note: y is already passed as the first argument
    - max_nb_epoch = maximum number of training epochs permitted
    - batch_size = number of samples per batch
    - patience = number of unhelpful epochs to tolerate
* Returns:
    - trained Keras model
'''
def train(model, X_train, y_train, X_val, y_val, process_X_data_func, 
    process_y_data_func, nb_features, nb_classes, process_X_data_func_args={}, 
    process_y_data_func_args={}, max_nb_epoch=100, batch_size=128, patience=1, 
    verbose=True):    
    # early stopping by monitoring validation loss
    # custom implementation to accomodate batch data processing & training
    class EarlyStopping(Exception): pass
    try:
        best_val_loss = sys.float_info.max # some really high initial value
        patience_counter = patience

        for epoch in range(1, max_nb_epoch + 1):
            epoch_start = time.time()

            if verbose: 
                print('\n', 'Epoch {} start:'.format(epoch))
                print('{} train batches'
                    .format(int(ceil(float(len(X_train)) / batch_size))))

            # train by batch
            for i, (X, y) in enumerate(izip(chunks(X_train, batch_size), 
                chunks(y_train, batch_size))):
                if i % 250 == 0 and verbose:
                    print('-- train batch {}'.format(i))

                assert len(X) == len(y) # chunk sizes should be equal
                w = len(X) # chunk size serves as weight when averaging
                X = process_X_data_func(X, **process_X_data_func_args)
                y = process_y_data_func(y, **process_y_data_func_args)

                model.train_on_batch(X, y)

            if verbose:
                print('{} val batches'
                    .format(int(ceil(float(len(X_val)) / batch_size))))

            # validation by batch
            val_losses, val_accs, val_weights = [], [], []
            for i, (X, y) in enumerate(izip(chunks(X_val, batch_size), 
                chunks(y_val, batch_size))):
                if i % 250 == 0 and verbose:
                    print('-- val batch {}'.format(i))

                assert len(X) == len(y) # chunk sizes should be equal
                w = len(X) # chunk size serves as weight when averaging
                X = process_X_data_func(X, **process_X_data_func_args)
                y = process_y_data_func(y, **process_y_data_func_args)

                batch_loss, batch_acc = model.test_on_batch(X, y)

                val_losses.append(batch_loss)
                val_accs.append(batch_acc)
                val_weights.append(w)

            val_loss = np.average(val_losses, weights=val_weights)
            val_acc = np.average(val_accs, weights=val_weights)
            if verbose:
                print('Epoch {} / loss: {:.3f} / acc: {:.3f} / time: {:.3f} s'
                    .format(epoch, val_loss, val_acc, time.time() - epoch_start))

            # trigger early stopping (do not save current model)
            if val_loss >= best_val_loss:
                if patience_counter == 0:
                    if verbose: print('Early stopping on epoch {}'.format(epoch))
                    raise EarlyStopping
                patience_counter -= 1
            # continue training, go onto next epoch
            else: 
                patience_counter = patience
                best_val_loss = val_loss
                best_epoch = epoch
                model_weights = model.get_weights() # save best model
        if verbose:
            print('Hit max number of training epochs at epoch {}'.format(epoch))
        raise EarlyStopping

    except EarlyStopping:
        if verbose: print('Best epoch was epoch {}'.format(best_epoch))
        # load most recent model weights from prior to early stopping
        model.set_weights(model_weights)
        return model

'''
* Tests a trained Keras model in a batch-wise fashion. It is too expensive to 
  vectorize the entire test set, so the data is vectorized in a streaming
  fashion as needed (e.g., in sync with training) using a generator.
* Expects:
    - model = trained Keras model
    - X_test, y_test, y_train = set of non-vectorized, encoded testing data
    - process_X_data_func = function to process feature data for modeling
    - process_y_data_func = function to process class data for modeling
    - nb_features = number of features
    - nb_classes = number of classes
    - process_X_data_func_args = additional arguments for process_X_data_func(); 
      note: X is already passed as the first argument
    - process_y_data_func_args = additional arguments for process_y_data_func(); 
      note: y is already passed as the first argument
    - batch_size = number of samples per batch
* Returns:
    - test loss
    - test accuracy
    - array of predicted probability vectors
'''
def test(model, X_test, y_test, process_X_data_func, process_y_data_func, 
    nb_features, nb_classes, process_X_data_func_args={},
    process_y_data_func_args={}, batch_size=512, verbose=True):

    if verbose: 
        print('{} test batches'
            .format(int(ceil(float(len(X_test)) / batch_size))))

    test_probas = np.empty([0, nb_classes])

    # testing by batch
    test_losses, test_accs, test_weights = [], [], []

    for i, (X, y) in enumerate(izip(chunks(X_test, batch_size), 
        chunks(y_test, batch_size))):
        if i % 250 == 0 and verbose: 
            print('-- test batch {}'.format(i))

        assert len(X) == len(y) # chunk sizes should be equal
        w = len(X) # chunk size serves as weight when averaging
        X = process_X_data_func(X, **process_X_data_func_args)
        y = process_y_data_func(y, **process_y_data_func_args)

        batch_loss, batch_acc = model.test_on_batch(X, y)
        batch_probas = model.predict_proba(X, batch_size=batch_size, 
            verbose=0)

        test_losses.append(batch_loss)
        test_accs.append(batch_acc)
        test_probas = np.append(test_probas, batch_probas, axis=0)
        test_weights.append(w)

    test_loss = np.average(test_losses, weights=test_weights)
    test_acc = np.average(test_accs, weights=test_weights)

    if verbose:
        print('Final test loss: {:5f} / accuracy: {:.15f}'
            .format(test_loss, test_acc))
        print()
    
    return (test_loss, test_acc), test_probas

'''
* Saves test results (softmax output, true label) to a file.
* Expects:
    - y_test_probas = 2D array of predicted probabilities; rows (first index) 
      represent samples, columns (second index) represent class indices
    - y_test = list of true classes
    - path = filepath where the file is saved
'''
def save_test_results(y_test_probas, y_test, path):
    with open(path, 'w+') as f:
        for prob_vector, target in zip(y_test_probas, y_test):
            for idx, p in enumerate(prob_vector):
                if idx != 0: f.write('\t')
                f.write(str(p))
            f.write('\t')
            f.write(str(target))
            f.write('\n')

'''
* Gets an array of probability predictions from some input feature data, 
  using the given Keras model.
* Expects:
    - model = compiled & trained Keras model
    - X_test = features of non-vectorized, encoded testing data
    - process_X_data_func = function to process feature data
    - process_X_data_func_args = additional arguments for process_X_data_func(); 
      note: X is already passed as the first argument
* Returns:
    - 2D array of predicted probabilities; rows (first index) represent samples, 
      columns (second index) represent class indices
'''
def predict_proba(model, X_test, process_X_data_func=None, 
    process_X_data_func_args={}):
    if process_X_data_func is not None:
        X_test = process_X_data_func(X_test, **process_X_data_func_args)

    return model.predict_proba(X_test)

'''
* Converts an array of probability predictions to predictions (index
  of predicted class), by choosing the class with highest probability.
* Expects:
    - probas = 2D array of predicted probabilities; rows (first index) 
      represent samples, columns (second index) represent class indices
* Returns:
    - vector of integer indices of predicted classes
'''
def probas_to_preds(probas):
    return np.argmax(probas, axis=1)

'''
* Gets an array of predictions (index of predicted class) from some 
  input feature data, using the given Keras model.
* Expects:
    - model = trained Keras model
    - X_test = features of non-vectorized, encoded testing data
    - process_X_data_func = function to process feature data
    - process_X_data_func_args = additional arguments for process_X_data_func(); 
      note: X is already passed as the first argument
* Returns:
    - vector of integer indices of predicted classes
'''
def predict(model, X, process_X_data_func=None, process_X_data_func_args={}):
    probas = predict_probas(model, X, process_X_data_func, 
        process_X_data_func_args=process_X_data_func_args)
    return probas_to_preds(probas)

'''
* Saves to file a compiled Keras model via HDF5. Requires HDF5 along with
  the Python library h5py to be installed. 
* Expects:
    - model = compiled Keras model
    - path = filepath where model should be saved
'''
def save_model(model, path):
    model.save(path)

'''
* Loads from file a saved Keras model via HDF5. (Probably) requires HDF5 along
  with the Python library h5py to be installed. 
* Expects:
    - path = filepath where model should be loaded from
* Returns:
    - compiled Keras model
'''
def load_model(path):
    return keras_load_model(path)
