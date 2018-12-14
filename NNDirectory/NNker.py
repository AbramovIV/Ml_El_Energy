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

ls = LearningSet(path_to_df=r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt')
my_df = ls.create_learningSet(ls.initial_df)
my_df = my_df.drop('HistoryLoad', axis=1)
my_df = my_df.loc[my_df['Year'] < 2017]
response = 'DiffHistoryLoad'


df_enc = ls.encode_categorials_features(my_df)
df_scale = ls.my_scale(df_enc)

#cor = df_scale.corr()
#plt.matshow(cor)
#plt.xticks(range(len(df_scale.columns)), df_scale.columns)
#plt.yticks(range(len(df_scale.columns)), df_scale.columns)
#plt.colorbar()
#plt.show()

Y = df_scale[response].values
X = df_scale.drop([response, 'Id'], axis=1).values
input_dim = X.shape[1]
print(input_dim)


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(26, input_dim=input_dim,
                    kernel_initializer='random_normal',
                    bias_initializer='zeros',
                    activation='relu'
                   ))
    model.add(Dense(1, activation='linear'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adadelta(lr=0.1), metrics=['mae'])
    return model


my_cv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in my_cv.split(X):
    X_train_cv = X[train_idx]
    y_train_cv = Y[train_idx]
    X_valid_cv = X[val_idx]
    y_valid_cv = Y[val_idx]

my_batch_size = 168
nn_model = create_model()

earlyStopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=0, mode='auto')

history = nn_model.fit(x=X_train_cv,
                       y=y_train_cv,
                       epochs=400,
                       shuffle=False,
                       batch_size = my_batch_size,
                       verbose=2,
                       validation_data=(X_valid_cv, y_valid_cv),
                       callbacks=[earlyStopping]
                       )

plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()



