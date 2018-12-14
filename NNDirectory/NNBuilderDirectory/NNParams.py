from typing import List
import typing

from keras.constraints import max_norm
from xlwings import xrange


class BaseNNParams:
    optimizer = None
    l1reg = float
    l2reg = float
    maxEpoch_gridsearch = int
    activation = str
    input_dim = int
    loss = str

    def __init__(self, optimizer, l1reg, l2reg, maxEpoch_gridsearch, activation, input_dim, loss):
        self.optimizer = optimizer
        self.l1reg = l1reg
        self.l2reg = l2reg
        self.maxEpoch_gridsearch = maxEpoch_gridsearch
        self.activation = activation
        self.input_dim = input_dim
        self.loss = loss

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
from keras.optimizers import SGD
import pandas as pd
import keras.backend as K
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from NNDirectory.LearningSetCl import LearningSet
from NNDirectory.MyCallBacks import myCallBacks


class NNparams(BaseNNParams):
    hidden = List[int]
    dropout = List[int]
    lr_metric = None
    batch_size = None

    def __init__(self, hidden, dropout, optimizer, l1reg, l2reg, maxEpoch_gridsearch, activation, input_dim,
                 loss, batch_size):
        BaseNNParams.__init__(self, optimizer=optimizer, l1reg= l1reg, l2reg=l2reg,
                              maxEpoch_gridsearch=maxEpoch_gridsearch,
                              activation=activation, input_dim=input_dim, loss=loss)
        self.hidden = hidden
        self.dropout = dropout
        self.batch_size = batch_size

    def buildNNModel(self):
        model = Sequential()
        for i in range(0, len(self.hidden)):
            if i == 0:
                model.add(Dense(self.hidden[i], input_dim=self.input_dim,
                                kernel_initializer='random_uniform',
                                bias_initializer='zeros',
                                activation=self.activation,
                                activity_regularizer=keras.regularizers.l1_l2(self.l1reg, self.l2reg)
                                ))
                #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.001))
                model.add(Dropout(self.dropout[i]))
            if i != 0:
                model.add(Dense(self.hidden[i],
                                kernel_initializer='random_normal',
                                bias_initializer='zeros',
                                activation=self.activation,
                                activity_regularizer=keras.regularizers.l1_l2(self.l1reg, self.l2reg)
                                ))
                #model.add(keras.layers.advanced_activations.LeakyReLU(alpha=0.001))
                model.add(Dropout(self.dropout[i]))
        model.add(Dense(1, activation='linear'))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['mae'])
        return model




