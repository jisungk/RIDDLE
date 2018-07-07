"""mlp.py

Deep MLP model conforming to the RIDDLE API.

Requires:   Keras (and its dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2018, all rights reserved
"""

from __future__ import print_function
from functools import partial

from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.layers.core import Activation
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from .model import Model


class MLP(Model):
    """Deep MLP RIDDLE model."""

    def __init__(self, num_feature, num_class, num_hidden_layer=1,
                 num_hidden_node=512, dropout=0.5, activation='prelu',
                 learning_rate=0.02, **kwargs):
        """Initialize a deep MLP.

        Arguments:
            num_feature: int
                number of features
            num_class: int
                number of classes
            num_hidden_layer: int
                number of hidden layers
            dropout: float
                dropout rate; values in [0, 1)
            activation: string
                name of activation function, see https://keras.io/activations/
            learning_rate: float
                learning rate
        """
        model = _create_mlp_model(
            num_feature, num_class, num_hidden_layer=num_hidden_layer,
            num_hidden_node=num_hidden_node, dropout=dropout,
            activation=activation, learning_rate=learning_rate)

        super(MLP, self).__init__(model, num_class, **kwargs)

        self.process_x = partial(_process_x, num_feature=num_feature)
        self.process_y = partial(_process_y, num_class=num_class)


def _process_x(x, num_feature):
    """Vectorize feature data via binary encoding."""
    assert num_feature > 0
    tokenizer = Tokenizer(num_words=num_feature)
    return tokenizer.sequences_to_matrix(x, mode='binary')


def _process_y(y, num_class):
    """Vectorize class labels via one-hot encoding."""
    assert num_class > 0
    return np_utils.to_categorical(y, num_class)


def _create_mlp_model(num_feature, num_class, num_hidden_layer,
                      num_hidden_node, dropout, activation, learning_rate):
    """Create a compiled Keras model.

    Arguments:
        num_feature: int
            number of features
        num_class: int
            number of classes
        num_hidden_layer: int
            number of hidden layers
        dropout: float
            dropout rate; values in [0, 1)
        activation: string
            name of activation function, see https://keras.io/activations/
        learning_rate: float
            learning rate

    Returns:
        model: keras.models.Sequential
            compiled Keras model
    """
    model = Sequential()

    if activation == 'prelu':
        activation_layer = PReLU
    else:
        activation_layer = lambda: Activation(activation)

    # input layer + first hidden layer
    model.add(Dense(num_hidden_node, input_shape=(num_feature,)))
    model.add(activation_layer())
    model.add(Dropout(dropout))

    for _ in range(num_hidden_layer):
        model.add(Dense(num_hidden_node))
        model.add(activation_layer())
        model.add(Dropout(dropout))

    # output layer
    model.add(Dense(num_class))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=learning_rate), metrics=['accuracy'])

    return model
