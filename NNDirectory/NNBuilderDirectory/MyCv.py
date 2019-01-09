import keras
from keras.backend import random_uniform
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from keras.models import model_from_json, load_model
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
        self.model_path = r'C:\Users\vgv\Desktop\PythonData\filepath="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"#'
        self.model_path_json = r'C:\Users\vgv\Desktop\PythonData\model.json'
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
                                                           min_delta=1e-4,
                                                           patience=25,
                                                           verbose=2,
                                                           mode='auto',
                                                           restore_best_weights=False
                                                           )

        self.early_stop_cv_final = keras.callbacks.EarlyStopping(
                                                                 monitor='mean_absolute_error',
                                                                 #monitor='loss',
                                                                 min_delta=1e-4,
                                                                 patience=5,
                                                                 verbose=2,
                                                                 mode='auto',
                                                                 restore_best_weights=False
                                                                 )



    def myCross_validation(self, nn: NNparams, X, Y, n_folds=5, trained=False, max_epoch=1000):
        my_cv = TimeSeriesSplit(n_splits=n_folds)
        self.results_cv = list()
        counter_load_weigths = 0

        #if trained is True:
        #    model = load_model(filepath=self.model_path, compile=True)
        #    nn.set_nn_model(model)

        for train_idx, val_idx in my_cv.split(X):
            x_train_cv = X[train_idx]
            y_train_cv = Y[train_idx]
            x_valid_cv = X[val_idx]
            y_valid_cv = Y[val_idx]
        #x_train_cv = X[ : (X.shape[0] -168), ]
        #y_train_cv = Y[ : (X.shape[0] -168), ]
        #x_valid_cv = X[(X.shape[0] - 167) :, ]
        #y_valid_cv = Y[(X.shape[0] - 167) :, ]
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
            train_error = res_fit.history['mean_absolute_error']
            valid_error = res_fit.history['val_mean_absolute_error']
            self.results_cv.append(([train_error], [valid_error]))
            #nn.get_nn_model().save(filepath=self.model_path, overwrite=True, include_optimizer=True)

        self.recommended_train_epoch = round(self.epoch_cv / n_folds)
        # load best weights
        # nn.get_nn_model().load_weights(self.model_cv_filepath)
        #nn.get_nn_model().save_weights(self.model_cv_filepath)
        #model = load_model(filepath=self.model_path, compile=True)
        #nn.set_nn_model(model)
        #self.save_model_to_json(nn)
        return nn.get_nn_model()

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

        # nn.get_nn_model().load_weights(self.model_cv_filepath)
        nn.get_nn_model().fit(x=X,
                              y=Y,
                              epochs=self.recommended_train_epoch,
                              shuffle=False,
                              batch_size=nn.batch_size,
                              verbose=2,
                              callbacks=[self.early_stop_cv_final]
                              )
        # load best weights
        # nn.get_nn_model().load_weights(self.model_cv__final_filepath)
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

