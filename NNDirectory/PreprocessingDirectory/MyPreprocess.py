import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from typing import Dict, List
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler

from DataDirectory.LoadData import LoadData
from NNDirectory.LsDirectory.LearningSetCl import LearningSet


class MyPreprocess:
    lags_dict_diff = Dict[int, int]
    prediction_lag_diff = int
    initital_df = pd.DataFrame
    response = str
    learning_set = pd.DataFrame
    categorial_cols = List[str]
    target = str

    def __init__(self, initital_df: pd.DataFrame, target: str, response: str,
                 prediction_lag_diff: int, lags_dict_diff: Dict[int, int]):
        self.initital_df = initital_df
        self.prediction_lag_diff = prediction_lag_diff
        self.target = target
        self.response = response
        self.lags_dict_diff = lags_dict_diff
        self.learning_set = initital_df.copy(deep=True)

    def set_prediction_series(self):
        response_series = self.initital_df[self.target]
        response_series = response_series.diff(self.prediction_lag_diff).values
        self.learning_set[response] = pd.Series(response_series, index=self.learning_set.index)

    def set_augmentations_lags(self, isDiff:bool):
        init_series = self.learning_set[self.response]
        if isDiff is False:
            for lag in self.lags_dict_diff.keys():
                lag_series = init_series.shift(lag).values
                name = 'lag_' + str(lag)
                self.learning_set[name] = pd.Series(lag_series, index=self.learning_set.index)
        else:
            for lag in self.lags_dict_diff.keys():
                diff_key = self.lags_dict_diff.get(lag)
                diff_series = init_series.diff(diff_key)
                lag_series = diff_series.shift(lag).values
                name = 'lag_' + str(lag)
                self.learning_set[name] = pd.Series(lag_series, index=self.learning_set.index)

    def encode_one_hot(self, df: pd.DataFrame, colls_to_one_hot: List[str]):
        df_c = df.copy(deep=True)

        for onh in colls_to_one_hot:
            dummies = pd.get_dummies(df_c[onh], prefix=onh, drop_first=False)
            df_c = pd.concat([df_c, dummies], axis=1)
        df_c = df_c.drop(colls_to_one_hot, axis=1)
        return df_c

    def scale_df(self, df: pd.DataFrame):
        df_c = df.copy(deep=True)
        old_indexis = df_c.index.values
        x = df_c.values
        self.min_max_scaler = MinMaxScaler()
        x_scaled = self.min_max_scaler.fit_transform(x)
        res = pd.DataFrame(x_scaled, columns=df_c.columns.values, index=old_indexis)
        res[response].iloc[-1] = np.nan
        return res

load_data = LoadData(r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt')
response = 'DiffHistoryLoad'
my_df = load_data.initDf

lags_dict = {1: 0, 2: 0, 3: 0, 4: 0, 24: 0
             5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
             15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
             24: 0, 48: 0, 72: 0, 96: 0, 120: 0, 144: 0, 168: 0,
             # 25: 24, 47: 24, 73: 24, 97: 24, 121: 24, 145: 24, 169: 24
             # 25: 24, 49: 24, 73: 24, 97: 24, 121: 24, 145: 24, 169: 24,
             }

preprocess = MyPreprocess(initital_df=my_df, target='HistoryLoad', response='DiffHistoryLoad',
                          prediction_lag_diff=1, lags_dict_diff=lags_dict)
preprocess.set_prediction_series()

#year live as numeric feature
categorial_cols = ['Month', 'Day', 'DayName', 'WorkType', 'Time']
preprocess.learning_set = preprocess.encode_one_hot(preprocess.learning_set, categorial_cols)
preprocess.set_augmentations_lags(isDiff=False)



input_shape = len(preprocess.learning_set.columns.values) - 3 #i.e. - {HistoryLoad, DiffHistoryLoad, Id}
perc = 90
hid = [len(numcols)*3]  #round( (len(numcols)*perc)/100)
drop = [0.5]
# create neural model
nn = NNparams(hidden=hid, dropout=drop,
              optimizer=keras.optimizers.Adam(amsgrad=True),
              l1reg=0, l2reg=0,
              activation='relu', input_dim=input_shape,
              loss='mean_squared_error',
              train_metric=['mean_absolute_error'],
              batch_size=168,
              kernel_init='random_uniform', bias_init='zeros',
              compile=True
              )






