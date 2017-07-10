"""
deep_mlp.py

Everything you need to train a deep multilayer perceptron (deep MLP) using
the RIDDLE library. 

Requires:   NumPy, Keras (and their dependencies)

Author:     Ji-Sung Kim, Rzhetsky Lab
Copyright:  2016, all rights reserved
"""

from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation 
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
from keras.utils import np_utils  
from keras.preprocessing.text import Tokenizer  

# -------------------------------- FUNCTIONS --------------------------------- #

'''
* Provides the architecture for a deep MLP model as a compiled Keras model.
* Expects:
    - nb_features = number of features
    - nb_classes = number of classes
    - learning_rate = learning rate
* Returns:
    - Keras Sequential model configuration as a JSON string 
'''
def create_base_model(nb_features, nb_classes, learning_rate=0.02):
    model = Sequential() 

    # input layer + first hidden layer 
    model.add(Dense(512, kernel_initializer='lecun_uniform', input_shape=(nb_features,)))
    model.add(PReLU()) 
    model.add(Dropout(0.5)) 

    # additional hidden layer
    model.add(Dense(512, kernel_initializer='lecun_uniform')) 
    model.add(PReLU()) 
    model.add(Dropout(0.75)) 
 
    # output layer 
    model.add(Dense(nb_classes, kernel_initializer='lecun_uniform')) 
    model.add(Activation('softmax')) 

    model.compile(loss='categorical_crossentropy', 
        optimizer=Adam(lr=learning_rate), metrics=['accuracy'])  

    return model

'''
* Processes feature data (X) for input to a deep MLP via binary encoding. 
* Expects:
    - X = feature data as a list of list of feature indices
        e.g., [[1, 3, 4], [4, 2, 5, 1, 3], [0]]
    - nb_features = number of features
* Return:
    - X represented as a binary array with dimensions (nb_sample, nb_features)
'''
def process_X_data(X, nb_features):
    assert nb_features > 0

    tokenizer = Tokenizer(num_words=nb_features)
    return tokenizer.sequences_to_matrix(X, mode='binary')

'''
* Processes class data (y) for input to a deep MLP via one-hot encoding. 
* Expects:
    - y = class data as a list of class indices
        e.g., [1, 0, 3]
    - nb_classes = number of classes
* Return:
    - y represented as a binary array with dimension (nb_sample, nb_class) 
'''
def process_y_data(y, nb_classes):
    assert nb_classes > 0

    return np_utils.to_categorical(y, nb_classes)