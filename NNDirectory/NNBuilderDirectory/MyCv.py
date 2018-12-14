from typing import List

import keras
import pandas as pd
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
import numpy as np
from NNDirectory.LearningSetCl import LearningSet
from NNDirectory.MyCallBacks import myCallBacks
from NNDirectory.NNBuilderDirectory.NNParams import NNparams
import matplotlib.pyplot as plt

class MyCv:
    max_epoch = int
    results_cv = None
    epoch_cv = 0
    model_filepath = None
    checkpoint_cv = None
    checkpoint = None
    counter_load_weigths = 0
    def __init__(self, model_filepath):
        self.model_filepath = model_filepath
        self.checkpoint_cv = ModelCheckpoint(self.model_filepath, monitor='val_mean_absolute_error', verbose=1,
                                          save_best_only=True,
                                          mode='min')
        self.checkpoint = ModelCheckpoint(self.model_filepath, monitor='mean_absolute_error', verbose=1,
                                          save_best_only=True,
                                          mode='min')

    def myCross_validation(self, nn_model, X, Y, n_folds, max_epoch, batch_size, early_stop):
        my_cv = TimeSeriesSplit(n_splits=n_folds)
        self.results_cv = list()

        for train_idx, val_idx in my_cv.split(X):

            if self.counter_load_weigths > 0:
               nn_model.load_weights(self.model_filepath)

            X_train_cv = X[train_idx]
            y_train_cv = Y[train_idx]
            X_valid_cv = X[val_idx]
            y_valid_cv = Y[val_idx]

            res_fit = nn_model.fit(x=X_train_cv,
                                   y=y_train_cv,
                                   epochs=max_epoch,
                                   shuffle=False,
                                   batch_size=batch_size,
                                   verbose=2,
                                   validation_data=(X_valid_cv, y_valid_cv),
                                   callbacks=[early_stop, self.checkpoint_cv]
                                   )
            self.counter_load_weigths = self.counter_load_weigths + 1
            self.epoch_cv = self.epoch_cv + len(res_fit.history['loss'])
            train_error = res_fit.history['mean_absolute_error']
            valid_error = res_fit.history['val_mean_absolute_error']
            self.results_cv.append(([train_error], [valid_error]))
        self.epoch_cv = round(self.epoch_cv / n_folds)
        # load best weights
        nn_model.load_weights(self.model_filepath)
        return nn_model

    def final_model(self, nn_model, X, Y, batch_size, early_stop):
        nn_model.load_weights(self.model_filepath)
        nn_model.fit(x=X,
                     y=Y,
                     epochs=self.epoch_cv,
                     shuffle=False,
                     batch_size=batch_size,
                     verbose=2,
                     callbacks=[early_stop, self.checkpoint]
                     )
        nn_model.load_weights(self.model_filepath)

        #print('validation_loss = ', nn_model.metrics_names[1])
        return nn_model

    def buildCvPlot(self):
        list_train = list()
        list_valid = list()
        for i in range(0, len(self.results_cv)):
            list_train.extend(self.results_cv[i][0][0])
            list_valid.extend(self.results_cv[i][1][0])
        x = list(range(0, len(list_train)))
        plt.plot(x, list_train, 'b-.', x, list_valid, 'r-.')
        plt.show()

