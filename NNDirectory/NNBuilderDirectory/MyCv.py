import keras
from keras.backend import random_uniform
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

from NNDirectory.NNBuilderDirectory.NNParams import NNparams


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
    def __init__(self, model_cv_filepath, model_cv__final_filepath, path_to_initial_weigths):
        self.model_cv_filepath = model_cv_filepath
        self.model_cv__final_filepath = model_cv__final_filepath
        self.model_initial_weights = path_to_initial_weigths
        self.checkpoint_cv = ModelCheckpoint(self.model_cv_filepath,
                                             #monitor='val_mean_absolute_error', verbose=1,
                                             monitor='val_loss', verbose=1,
                                             save_best_only=True,
                                             mode='min'
                                             )

        self.checkpoint_final_cv = ModelCheckpoint(self.model_cv__final_filepath,
                                                   monitor='loss', verbose=1,
                                                   save_best_only=True,
                                                   mode='min'
                                                   )

        self.early_stop_cv = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           min_delta=1e-4,
                                                           patience=15,
                                                           verbose=2,
                                                           mode='auto'
                                                           )

        self.early_stop_cv_final = keras.callbacks.EarlyStopping(monitor='loss',
                                                                 min_delta=1e-4,
                                                                 patience=5,
                                                                 verbose=2,
                                                                 mode='auto'
                                                                 )

    #def myCross_validation(self, nn: NNparams, X, Y, n_folds=3, max_epoch=1000, trained=False):
    #    if trained is True:
    #        nn.get_nn_model().load_weights(self.model_cv__final_filepath)
#
    #    my_cv = TimeSeriesSplit(n_splits=n_folds)
    #    self.results_cv = list()
    #    counter_load_weigths = 0
#
    #    x_train_cv = X[:-336]
    #    y_train_cv = Y[:-336]
    #    x_valid_cv = X[337:]
    #    y_valid_cv = Y[337:]
#
    #    res_fit = nn.get_nn_model().fit(x=x_train_cv,
    #                                    y=y_train_cv,
    #                                    epochs=max_epoch,
    #                                    shuffle=False,
    #                                    batch_size=nn.batch_size,
    #                                    verbose=2,
    #                                    validation_data=(x_valid_cv, y_valid_cv),
    #                                    callbacks=[self.early_stop_cv, self.checkpoint_cv]
    #                                    )
    #    counter_load_weigths = counter_load_weigths + 1
    #    self.epoch_cv = self.epoch_cv + len(res_fit.history['loss'])
    #    #train_error = res_fit.history['mean_absolute_error']
    #    train_error = res_fit.history['loss']
    #    #valid_error = res_fit.history['val_mean_absolute_error']
    #    valid_error = res_fit.history['val_loss']
    #    self.results_cv.append(([train_error], [valid_error]))
    #    self.recommended_train_epoch = self.epoch_cv #/ nfolds#
    #    # load best weights
    #    nn.get_nn_model().load_weights(self.model_cv_filepath)
    #    return nn.get_nn_model()

    def myCross_validation(self, nn: NNparams, X, Y, n_folds=3, trained=False, max_epoch=1000):
        my_cv = TimeSeriesSplit(n_splits=n_folds)
        self.results_cv = list()
        counter_load_weigths = 0

        for train_idx, val_idx in my_cv.split(X):
            if counter_load_weigths > 0:
                nn.get_nn_model().load_weights(self.model_cv_filepath)
            x_train_cv = X[train_idx]
            y_train_cv = Y[train_idx]
            x_valid_cv = X[val_idx]
            y_valid_cv = Y[val_idx]
            res_fit = nn.get_nn_model().fit(x=x_train_cv,
                                            y=y_train_cv,
                                            epochs=max_epoch,
                                            shuffle=False,
                                            batch_size=nn.batch_size,
                                            verbose=2,
                                            validation_data=(x_valid_cv, y_valid_cv),
                                            callbacks=[self.early_stop_cv, self.checkpoint_cv]
                                            )
            counter_load_weigths = counter_load_weigths + 1
            self.epoch_cv = self.epoch_cv + len(res_fit.history['loss'])
            train_error = res_fit.history['mean_absolute_error']
            valid_error = res_fit.history['val_mean_absolute_error']
            self.results_cv.append(([train_error], [valid_error]))
        self.recommended_train_epoch = round(self.epoch_cv / n_folds)
        # load best weights
        nn.get_nn_model().load_weights(self.model_cv_filepath)
        return nn.get_nn_model()

    def train_final_model(self, nn : NNparams, X, Y):

        nn.get_nn_model().load_weights(self.model_cv_filepath)
        nn.get_nn_model().fit(x=X,
                              y=Y,
                              epochs=self.recommended_train_epoch,
                              shuffle=False,
                              batch_size=nn.batch_size,
                              verbose=2,
                              callbacks=[self.early_stop_cv_final, self.checkpoint_final_cv]
                              )
        # load best weights
        nn.get_nn_model().load_weights(self.model_cv__final_filepath)
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
        plt.axes().set_ylabel('Loss')
        plt.show()

