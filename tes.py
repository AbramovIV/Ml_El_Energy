import keras
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from DataDirectory.LoadData import LoadData
from NNDirectory.NNBuilderDirectory.MyCv import MyCv
from NNDirectory.NNBuilderDirectory.NNParams import NNparams
from NNDirectory.PreprocessingDirectory.MyMinMaxScaller import MyMinMaxScaller
from NNDirectory.PreprocessingDirectory.MyPreprocess import MyPreprocess
from NNDirectory.PreprocessingDirectory.MyTestModel import MyTestModel

load_data = LoadData(r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt')
response = 'DiffHistoryLoad'
my_df = load_data.initDf

lags_dict = {1: 0, 2: 24, 3: 24, 4: 24,
             5: 24, 6: 24, 7: 24, 8: 24, 9: 24, 10: 24, 11: 24, # 12: 24, 13: 24, 14: 24
             # 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
             24: 24, 48: 24, 72: 24, 96: 24, 120: 24, 144: 24 # , 168: 0
             # 25: 24, 47: 24, 73: 24, 97: 24, 121: 24, 145: 24, 169: 24
             # 25: 24, 49: 24, 73: 24, 97: 24, 121: 24, 145: 24, 169: 24,
             }

preprocess = MyPreprocess(initital_df=my_df, target='HistoryLoad', response='DiffHistoryLoad',
                          prediction_lag_diff=1, lags_dict_diff=lags_dict)
preprocess.set_prediction_series()

# year live as numeric feature and filling learning set
categorial_cols = ['Month', 'DayName', 'WorkType', 'Time'] # 'Day'
preprocess.learning_set = preprocess.encode_one_hot(preprocess.learning_set, categorial_cols)
preprocess.set_augmentations_lags(isDiff=True)

# remove rows that contains NA values
preprocess.learning_set = preprocess.learning_set.dropna()

# reset indexes in learning set
preprocess.learning_set = preprocess.learning_set.reset_index(drop=True)

# input shape for input layer of NN
input_shape = len(preprocess.learning_set.columns.values) - 3 # i.e. - {HistoryLoad, DiffHistoryLoad, Id}

# number of features
num_of_predictors = preprocess.learning_set.shape[1] - preprocess.num_of_categorials + len(categorial_cols)

# num of neurons in hidden layer
hid = [num_of_predictors*3, num_of_predictors*2]

# dropout rate
drop = [0.5, 0.5]

# create neural model
nn = NNparams(hidden=hid, dropout=drop,
              optimizer=keras.optimizers.Adam(amsgrad=True),
              l1reg=1e-4, l2reg=1e-4,
              activation='tanh', input_dim=input_shape,
              loss='mean_squared_error',
              train_metric=['mean_absolute_error'],
              batch_size=168,
              kernel_init='random_uniform', bias_init='zeros',
              compile=True
              )

# cross validation
cross_val = MyCv(model_cv_filepath=r'C:\Users\vgv\Desktop\PythonData\cv_weights.hdf5',
                 model_cv__final_filepath=r'C:\Users\vgv\Desktop\PythonData\cv_final_weights.hdf5',
                 path_to_initial_weigths=r'C:\Users\vgv\Desktop\PythonData\init_weigths.hdf5',
                 hidden=hid,
                 inp_shape=input_shape
                 )

first_pred_df = preprocess.learning_set.loc[preprocess.learning_set['Year'] == 2017 ]
first_id = first_pred_df.iloc[0].Id
last_pred_id = my_df.iloc[-1].Id


test_model = MyTestModel(nn=nn, cross_val=cross_val, preprocess=preprocess)
test_model.test_my_model(int(first_id), int(last_pred_id))





