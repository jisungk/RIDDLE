"""model.py

Base RIDDLE classification model

Requires:   Keras (and its dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

from __future__ import print_function

from math import ceil
import time

from keras.models import load_model as keras_load_model
import numpy as np
from sklearn.metrics import log_loss

MAX_NUM_EPOCH = 100


class Model(object):
    """Base RIDDLE model."""

    def __init__(self, model, num_class, max_num_epoch=-1, batch_size=128,
                 patience=1):
        """Initialize a RIDDLE model.

        Arguments:
            model: keras.models.Model
                compiled Keras model
            num_class: int
                number of classes
            max_num_epoch: int
                maximum number of epochs
            batch_size: int
                batch size
            patience: int
                number of epochs permitted to demonstrate worse validation loss
                before early stopping is triggered
        """
        self._model = model
        self._num_class = num_class

        if max_num_epoch < 1:
            max_num_epoch = MAX_NUM_EPOCH
        self._max_num_epoch = max_num_epoch
        self._batch_size = batch_size
        self._patience = patience

    def train(self, x_train_unvec, y_train, x_val_unvec, y_val, verbose=True):
        """Train a model in a batch-wise fashion.

        Applies vectorization batch-by-batch to avoid memory overflow.

        Arguments:
            x_train_unvec: [[int]]
                training data feature indices that have not been vectorized; each
                inner list collects the indices of features that are present
                (binary on) for a sample
            y_train: [int]
                list of training class labels
            x_val_unvec: [[int]]
                validation data feature indices that have not been vectorized; each
                inner list collects the indices of features that are present
                (binary on) for a sample
            y_val: [int]
                list of validation class labels

            verbose: bool
                controls logging

        Returns:
            model: keras.models.Sequential
                trained Keras model
        """
        # early stopping by monitoring validation loss
        # custom implementation to accomodate batch data processing & training
        class EarlyStopping(Exception):
            pass
        try:
            best_val_loss = float('inf')
            patience_counter = self._patience

            for epoch in range(1, self._max_num_epoch + 1):
                epoch_start = time.time()

                if verbose:
                    print('\n', 'Epoch {} start:'.format(epoch))
                    print('{} train batches'.format(
                        int(ceil(float(len(x_train_unvec)) / self._batch_size))))

                # train by batch
                for i, (x, y) in enumerate(
                    zip(chunks(x_train_unvec, self._batch_size),
                        chunks(y_train, self._batch_size))):
                    if i % 250 == 0 and verbose:
                        print('-- train batch {}'.format(i))

                    assert len(x) == len(y)  # chunk sizes should be equal
                    x = self.process_x(x)
                    y = self.process_y(y)

                    self._model.train_on_batch(x, y)

                if verbose:
                    print('{} val batches'.format(
                        int(ceil(float(len(x_val_unvec)) / self._batch_size))))

                # validation by batch
                y_val_probas = np.empty([0, self._num_class])
                for i, (x, y) in enumerate(
                    zip(chunks(x_val_unvec, self._batch_size),
                        chunks(y_val, self._batch_size))):
                    if i % 250 == 0 and verbose:
                        print('-- val batch {}'.format(i))

                    assert len(x) == len(y)  # chunk sizes should be equal
                    x = self.process_x(x)
                    y = self.process_y(y)

                    batch_probas = self._model.predict_proba(
                        x, batch_size=self._batch_size, verbose=0)
                    y_val_probas = np.append(
                        y_val_probas, batch_probas, axis=0)

                val_loss = log_loss(y_val, y_val_probas,
                                    labels=range(self._num_class))

                if verbose:
                    print('Epoch {} / loss: {:.3f} / time: {:.3f} s'
                          .format(epoch, val_loss, time.time() - epoch_start))

                # trigger early stopping (do not save current model)
                if val_loss >= best_val_loss:
                    if patience_counter == 0:
                        if verbose:
                            print('Early stopping on epoch {}'.format(epoch))
                        raise EarlyStopping
                    patience_counter -= 1
                # continue training, go onto next epoch
                else:
                    patience_counter = self._patience
                    best_val_loss = val_loss
                    best_epoch = epoch
                    model_weights = self._model.get_weights()  # save best model

            if verbose:
                print('Hit max number of training epochs: {}'
                      .format(self._max_num_epoch))
            raise EarlyStopping

        except EarlyStopping:
            if verbose:
                print('Best epoch was epoch {}'.format(best_epoch))
            # load most recent model weights from prior to early stopping
            self._model.set_weights(model_weights)

    def predict_proba(self, x_test_unvec, verbose=True):
        """Predict class label probabilities in a batch-wise fashion.

        Applies vectorization batch-by-batch to avoid memory overflow.

        Arguments:
            x_test_unvec: [[int]]
                test data feature indices that have not been vectorized; each
                inner list collects the indices of features that are present
                (binary on) for a sample
            verbose: bool
                controls logging

        Returns:
            y_test_probas: np.array
                array of predicted probabilities with shape
                (num_sample, num_class)
        """
        if verbose:
            print('{} test batches'.format(
                int(ceil(float(len(x_test_unvec)) / self._batch_size))))

        y_test_probas = np.empty([0, self._num_class])
        for i, x in enumerate(chunks(x_test_unvec, self._batch_size)):
            if i % 250 == 0 and verbose:
                print('-- test batch {}'.format(i))

            x = self.process_x(x)
            batch_probas = self._model.predict_proba(
                x, batch_size=self._batch_size, verbose=0)
            y_test_probas = np.append(y_test_probas, batch_probas, axis=0)

        return y_test_probas

    def predict(self, x_test_unvec, verbose=True):
        """Predict class labels in a batch-wise fashion.

        Applies vectorization batch-by-batch to avoid memory overflow.

        Arguments:
            x_test_unvec: [[int]]
                test data feature indices that have not been vectorized; each
                inner list collects the indices of features that are present
                (binary on) for a sample
            verbose: bool
                controls logging

        Returns:
            y_test: np.array
                array of predicted class labels with shape (num_sample, )
        """
        y_test_probas = self.predict_proba(x_test_unvec, verbose=verbose)
        return np.argmax(y_test_probas, axis=1)

    def save_model(self, path):
        """Save internal Keras model to file."""
        self._model.save(path)

    def load_model(self, path):
        """Load Keras model from file."""
        self._model = keras_load_model(path)


def chunks(lst, n):
    """Generator which yields n-sized chunks from a list."""
    assert n > 0
    for i in range(0, len(lst), n):
        chunk = lst[i:i + n]
        yield chunk
