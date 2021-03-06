# import keras
import numpy as np
import pandas as pd
from typing import Dict, List
import matplotlib.pyplot as plt
from DataDirectory.LoadData import LoadData
# from NNDirectory.NNBuilderDirectory.MyCv import MyCv
# from NNDirectory.NNBuilderDirectory.NNParams import NNparams
# from NNDirectory.PreprocessingDirectory.MyMinMaxScaller import MyMinMaxScaller
from NNDirectory.LsDirectory.Prepare_Ls import Prepare_Ls
from NNDirectory.NNBuilderDirectory.NNH2o import NNH2o
from NNDirectory.PreprocessingDirectory.MyPreprocess import MyPreprocess
# from NNDirectory.PreprocessingDirectory.MyTestModel import MyTestModel

load_data = LoadData(r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt')
response = 'DiffHistoryLoad'
my_df = load_data.initDf

lags_dict = {1: 0, 2: 0, 3: 0, 4: 0,
             5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0,  13: 0, 14: 0,
             15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
             24: 0, 48: 0, 72: 0, 96: 0, 120: 0, 144: 0, 168: 0,
             25: 0, 49: 0, 73: 0, 97: 0, 121: 0, 145: 0, 169: 0
             }

preprocess = MyPreprocess(initital_df=my_df, target='HistoryLoad', response='DiffHistoryLoad',
                          prediction_lag_diff=1, lags_dict_diff=lags_dict)
preprocess.set_prediction_series()
# preprocess.set_prev_temp_workType()
# year live as numeric feature and filling learning set
categorial_cols = ['Month', 'DayName', 'WorkType', 'Time']
preprocess.categorial_cols = categorial_cols
preprocess.colls_to_one_hot = ['Month', 'DayName', 'Time', 'WorkType']


# set lags
preprocess.set_augmentations_lags(isDiff=False)
# remove rows that contains NA values
preprocess.learning_set = preprocess.learning_set.dropna()


preprocess.learning_set = preprocess.encode_one_hot(preprocess.learning_set)

##############  THERE INIT_DF FOR LS
first_pred_df = preprocess.learning_set.loc[preprocess.learning_set['Year'] == 2016 ]
first_id = first_pred_df.iloc[0].Id
last_pred_id = my_df.iloc[-1].Id

# number of features
num_of_predictors = preprocess.learning_set.shape[1] - preprocess.num_of_categorials + len(categorial_cols)

predictors = list(preprocess.learning_set)
predictors.remove(response)
predictors.remove('Id')
predictors.remove('HistoryLoad')

# plt.figure(1)
# plt.hist(preprocess.learning_set['pca2'], bins=200)
# plt.figure(2)
# plt.hist(preprocess.learning_set['pca1'], bins=200)
#
# matching_pca = [s for s in preprocess.learning_set.columns.values if "pca" in s]
# matching_pca.append("DiffHistoryLoad")
# df_scale = preprocess.learning_set.ix[:, matching_pca]
# cor = df_scale.corr()
#
# plt.figure(3)
# plt.matshow(cor)
# plt.xticks(range(len(df_scale.columns)), df_scale.columns)
# plt.yticks(range(len(df_scale.columns)), df_scale.columns)
# plt.colorbar()
# plt.show()


nnh2o = NNH2o(input_neurons=num_of_predictors, response=response, predictors=predictors, df_init=preprocess.learning_set.copy(),
                 preprocess=preprocess,
                 prepare_ls=Prepare_Ls(categorial_cols=categorial_cols, one_hot_encoding_names=preprocess.colls_to_one_hot, response = response))

nnh2o.run_test(int(first_id), int(last_pred_id))

# def get_perc(per, num):
#     return round(abs(num*per)/100)
#
# # num of neurons in hidden layer
# # hid = [num_of_predictors, get_perc(70,num_of_predictors), get_perc(40,get_perc(70,num_of_predictors))]
# hid = [num_of_predictors*3, num_of_predictors*2, num_of_predictors]
# # dropout rate
# drop = [0.5, 0.5, 0.5]
#
# # create neural model
# nn = NNparams(hidden=hid, dropout=drop,
#               optimizer=keras.optimizers.Adadelta(),
#               l1reg=0, l2reg=0,
#               activation='tanh', input_dim=input_shape,
#               loss='mean_squared_error',
#               train_metric=['mean_absolute_error'],
#               batch_size=168,
#               kernel_init='random_uniform', bias_init='zeros',
#               compile=True
#               )
#
# # cross validation
# cross_val = MyCv(model_cv_filepath=r'C:\Users\vgv\Desktop\PythonData\cv_weights.hdf5',
#                  model_cv__final_filepath=r'C:\Users\vgv\Desktop\PythonData\cv_final_weights.hdf5',
#                  path_to_initial_weigths=r'C:\Users\vgv\Desktop\PythonData\init_weigths.hdf5',
#                  hidden=hid,
#                  inp_shape=input_shape
#                  )
#
#
#
#
# test_model = MyTestModel(nn=nn, cross_val=cross_val, preprocess=preprocess)
# test_model.test_my_model(int(first_id), int(last_pred_id))





