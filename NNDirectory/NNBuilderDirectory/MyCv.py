from typing import List

import keras
from keras.backend import random_uniform
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit, KFold, ShuffleSplit
import matplotlib.pyplot as plt
from keras.models import model_from_json, load_model
from NNDirectory.NNBuilderDirectory.NNParams import NNparams
import numpy as np
import  pandas as pd
from random import randint

class MyCv:
    results_cv = None
    epoch_cv = 0
    recommended_train_epoch = int
    checkpoint_cv = None
    checkpoint_final_cv = None
    early_stop_cv = None
    early_stop_cv_final = None
    model_cv_filepath = str
    model_cv__final_filepath = str
    model_initial_weights = str
    hid = List[int]
    input_shape = int
    nnPred = NNparams

    def __init__(self, model_cv_filepath, model_cv__final_filepath, path_to_initial_weigths, hidden, inp_shape):
        self.input_shape = inp_shape
        self.hid = hidden
        self.model_path = r'C:\Users\Ilya\Desktop\PythonData\model.h5"#'
        self.model_path_json = r'C:\Users\Ilya\Desktop\PythonData\model.json'
        self.model_cv_filepath = model_cv_filepath
        self.model_cv__final_filepath = model_cv__final_filepath
        self.model_initial_weights = path_to_initial_weigths
        self.checkpoint_cv = ModelCheckpoint(self.model_cv_filepath,
                                             monitor='val_mean_absolute_error',
                                             #monitor='val_loss',
                                             verbose=1,
                                             save_best_only=False,
                                             save_weights_only=False,
                                             period=3,
                                             mode='min'
                                             )

        self.checkpoint_final_cv = ModelCheckpoint(self.model_cv__final_filepath,
                                                   monitor='mean_absolute_error',
                                                   #monitor='loss',
                                                   verbose=1,
                                                   save_best_only=True,
                                                   save_weights_only=True,
                                                   mode='min'
                                                   )

        self.early_stop_cv = keras.callbacks.EarlyStopping(
                                                           monitor='val_mean_absolute_error',
                                                           #monitor='val_loss',
                                                           min_delta=1e-3,
                                                           patience=30,
                                                           verbose=2,
                                                           mode='min',
                                                           restore_best_weights=True
                                                           )

        self.early_stop_cv_final = keras.callbacks.EarlyStopping(
                                                                 monitor='val_mean_absolute_error',
                                                                 #monitor='loss',
                                                                 min_delta=1e-3,
                                                                 patience=20,
                                                                 verbose=2,
                                                                 mode='min',
                                                                 restore_best_weights=True
                                                                 )



    def myCross_validation(self, nn: NNparams, Xset : pd.DataFrame, Yset : pd.Series, trained, n_folds=5, max_epoch=1000):
        my_cv = TimeSeriesSplit(n_splits=7)
        #my_cv = KFold(n_splits=6)
        self.results_cv = list()
        counter_load_weigths = 0

        # if trained is True:
        #     nnPred = NNparams(hidden=[self.hid], dropout=[0.0],
        #                       optimizer=keras.optimizers.Adam(),
        #                       l1reg=0, l2reg=0,
        #                       activation='relu', input_dim=self.input_shape,
        #                       loss='mean_sqaured_error',
        #                       train_metric=['mean_absolute_error'],
        #                       batch_size=168,
        #                       kernel_init='random_normal', bias_init='zeros',
        #                       compile=False
        #                       )
        #     nnPred.get_nn_model().load_weights(self.model_cv_filepath)
        #     nnPred.get_nn_model().compile(loss=nnPred.loss, optimizer=nnPred.optimizer, metrics=nnPred.train_metric)
        #     nn = nnPred
        # initial_ep = 0
        # for train_idx, val_idx in my_cv.split(X):
        #     if counter_load_weigths > 0:
        #         nnPred = NNparams(hidden=[self.hid], dropout=[0.0],
        #                           optimizer=keras.optimizers.Adam(lr=1e-3),
        #                           l1reg=0, l2reg=0,
        #                           activation='relu', input_dim=self.input_shape,
        #                           loss='mean_squared_error',
        #                           train_metric=['mean_absolute_error'],
        #                           batch_size=nn.batch_size,
        #                           kernel_init='random_normal', bias_init='zeros',
        #                           compile=False
        #                           )
        #         nnPred.get_nn_model().load_weights(self.model_cv_filepath)
        #         nnPred.get_nn_model().compile(loss=nnPred.loss, optimizer=nnPred.optimizer, metrics=nnPred.train_metric)
        #         nn = nnPred
        #     x_train_cv = X[train_idx]
        #     y_train_cv = Y[train_idx]
        #     x_valid_cv = X[val_idx]
        #     y_valid_cv = Y[val_idx]
        #
        #     res_fit = nn.get_nn_model().fit(x=x_train_cv,
        #                                     y=y_train_cv,
        #                                     epochs=max_epoch,
        #                                     shuffle=False,
        #                                     batch_size=nn.batch_size,
        #                                     verbose=2,
        #                                     validation_data=(x_valid_cv, y_valid_cv),
        #                                     callbacks=[self.early_stop_cv],
        #                                     initial_epoch=initial_ep
        #                                     )
        #     counter_load_weigths = counter_load_weigths + 1
        #     self.epoch_cv = self.epoch_cv + len(res_fit.history['loss'])
        #     initial_ep = self.epoch_cv
        #     train_error = res_fit.history['mean_absolute_error']
        #     valid_error = res_fit.history['val_mean_absolute_error']
        #     self.results_cv.append(([train_error], [valid_error]))
        #     nn.get_nn_model().save_weights(filepath=self.model_cv_filepath)
        ind = Xset.index.values
        last = ind[-1]
        x_train_cv = Xset.iloc[ : (last - 8140), :]
        y_train_cv = Yset[ : (last - 8140) ]
        x_valid_cv = Xset.iloc[(last - 8139):, :]
        y_valid_cv = Yset[(last - 8139): ]



        res_fit = nn.get_nn_model().fit(x=x_train_cv,
                                        y=y_train_cv,
                                        epochs=max_epoch,
                                        shuffle=False,
                                        batch_size=nn.batch_size,
                                        verbose=2,
                                        validation_data=(x_valid_cv, y_valid_cv),
                                        callbacks=[self.early_stop_cv]
                                        )
        counter_load_weigths = counter_load_weigths + 1
        self.epoch_cv = self.epoch_cv + len(res_fit.history['loss'])
        # initial_ep = self.epoch_cv
        train_error = res_fit.history['mean_absolute_error']
        valid_error = res_fit.history['val_mean_absolute_error']
        self.results_cv.append(([train_error], [valid_error]))
        nn.get_nn_model().save_weights(filepath=self.model_cv_filepath)

        # counter_load_weigths = counter_load_weigths + 1
        # self.epoch_cv = len(res_fit.history['loss'])
        # train_error = res_fit.history['mean_absolute_error']
        # valid_error = res_fit.history['val_mean_absolute_error']
        # self.results_cv.append(([train_error], [valid_error]))
        # #nn.get_nn_model().save(filepath=self.model_path, overwrite=True, include_optimizer=True)
        self.buildCvPlot()
        self.recommended_train_epoch = round(self.epoch_cv/counter_load_weigths)
        nn.get_nn_model().save_weights(filepath=self.model_cv_filepath)
        return nn.get_nn_model()


    #def myCross_validation(self, nn: NNparams, X, Y, n_folds=5, trained=False, max_epoch=1000):
    #    my_cv = TimeSeriesSplit(n_splits=5)
    #    #my_cv = KFold(n_splits=5)
    #    self.results_cv = list()
    #    counter_load_weigths = 0
#
    #    #if trained is True:
    #    #    model = load_model(filepath=self.model_path, compile=True)
    #    #    nn.set_nn_model(model)
#
    #    for train_idx, val_idx in my_cv.split(X):
    #        x_train_cv = X[train_idx]
    #        y_train_cv = Y[train_idx]
    #        x_valid_cv = X[val_idx]
    #        y_valid_cv = Y[val_idx]
    #    #x_train_cv = X[ : (X.shape[0] -168), ]
    #    #y_train_cv = Y[ : (X.shape[0] -168), ]
    #    #x_valid_cv = X[(X.shape[0] - 167) :, ]
    #    #y_valid_cv = Y[(X.shape[0] - 167) :, ]
    #        res_fit = nn.get_nn_model().fit( x=x_train_cv,
    #                                         y=y_train_cv,
    #                                         epochs=max_epoch,
    #                                         shuffle=False,
    #                                         batch_size=nn.batch_size,
    #                                         verbose=2,
    #                                         validation_data=(x_valid_cv, y_valid_cv),
    #                                         callbacks=[self.early_stop_cv]
    #                                         )
    #        counter_load_weigths = counter_load_weigths + 1
    #        self.epoch_cv = self.epoch_cv + len(res_fit.history['loss'])
    #        train_error = res_fit.history['mean_absolute_error']
    #        valid_error = res_fit.history['val_mean_absolute_error']
    #        self.results_cv.append(([train_error], [valid_error]))
    #        #nn.get_nn_model().save(filepath=self.model_path, overwrite=True, include_optimizer=True)
    #
    #    self.recommended_train_epoch = round(self.epoch_cv / n_folds)
    #    return nn.get_nn_model()

    def save_model_to_json(self, myNN: NNparams):
        model_json = myNN.get_nn_model().to_json()
        with open(self.model_path_json, "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
            myNN.get_nn_model().save_weights(self.model_cv_filepath)
        print("Saved model to disk")

    def load_model_from_json(self):
        json_file = open(self.model_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        return  loaded_model
        print("Loaded model from disk")

    def train_final_model(self, nn : NNparams, X, Y):
        nnPred = NNparams(hidden=nn.hidden, dropout=nn.dropout,
                          optimizer=keras.optimizers.Adam(lr=1e-3, amsgrad=True),
                          l1reg=nn.l1, l2reg=nn.l2,
                          activation=nn.activation, input_dim=self.input_shape,
                          loss='mean_squared_error',
                          train_metric=['mean_absolute_error'],
                          batch_size=24,
                          kernel_init='random_uniform', bias_init='zeros',
                          compile=False
                         )
        nnPred.get_nn_model().load_weights(self.model_cv_filepath)
        nnPred.get_nn_model().compile(loss=nnPred.loss, optimizer=nnPred.optimizer, metrics=nnPred.train_metric)
        nn = nnPred
        ind = X.index.values
        last = ind[-1]
        x_train_cv = X.iloc[(last - 8139): , ]
        y_train_cv = Y[(last - 8139): ]

        x_train_cv = X.iloc[(last - 8139): (last - 1900), ]
        y_train_cv = Y[(last - 8139): (last - 1900) ]
        x_valid_cv = X.iloc[last - 1899 :, ]
        y_valid_cv = Y[last - 1899 : ]
                # create cv
        n = len(x_valid_cv)
        init_ind = x_valid_cv.index.values
        start_ind = init_ind[0]
        end_ind = init_ind[-1]
        rnd_ind = [randint(start_ind, end_ind) for p in range(start_ind, end_ind)]
        val_size = round(len(x_valid_cv) / 2)
        fin_val_ind = rnd_ind[:val_size]
        fin_x_val = x_valid_cv.iloc[x_valid_cv.index.isin(fin_val_ind)]
        fin_y_val = y_valid_cv[y_valid_cv.index.isin(fin_val_ind)]
        ind_to_train = set(init_ind) - set(fin_val_ind)
        to_train_x = x_valid_cv.iloc[x_valid_cv.index.isin(ind_to_train)]
        to_train_y = y_valid_cv[y_valid_cv.index.isin(ind_to_train)]
        x_train_cv = x_train_cv.append(to_train_x)
        y_train_cv = y_train_cv.append(to_train_y)

        res_fit = nn.get_nn_model().fit(x=x_train_cv,
                                        y=y_train_cv,
                                        epochs=10000,
                                        shuffle=False,
                                        batch_size=nn.batch_size,
                                        verbose=2,
                                        validation_data=(fin_x_val, fin_y_val),
                                        callbacks=[self.early_stop_cv_final],
                                        )
        nn.get_nn_model().save_weights(filepath=self.model_cv_filepath)
        # self.results_cv = list()
        train_error = res_fit.history['mean_absolute_error']
        valid_error = res_fit.history['val_mean_absolute_error']
        self.results_cv.append(([train_error], [valid_error]))
        #self.buildCvPlot()
        return nn.get_nn_model()

    def buildCvPlot(self):
        list_train = list()
        list_valid = list()
        for i in range(0, len(self.results_cv)):
            list_train.extend(self.results_cv[i][0][0])
            list_valid.extend(self.results_cv[i][1][0])
        x = list(range(0, len(list_train)))
        plt.plot(x, list_train, 'b-', x, list_valid, 'r-.')
        plt.axes().set_xlabel('epoch')
        plt.axes().set_ylabel('MAE')
        plt.show()

