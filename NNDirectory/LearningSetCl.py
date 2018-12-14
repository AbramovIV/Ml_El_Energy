from typing import Dict, Any, Union

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np

from NNDirectory.SupportFunctions import ordinal_transform_categorials, my_min_max_scaller, my_min_max_scaller_series, \
    one_hot_transform_categorials


class LearningSet:

    lags_dict = {1: 1, 2: 2, 3: 3, 4: 4,
                 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
                 24: 24, 48: 24, 72: 24, 96: 24, 120: 24, 144: 24, 168: 168
                }



    initial_df = None
    learning_set = None

    def __init__(self, path_to_df: str) -> None:
        self.initial_df = pd.read_table(path_to_df, sep=" ", header=0, na_values='nan', keep_default_na=True)
        self.initial_df['PrevTemp'] = self.initial_df['Temperature'].shift(24)
        self.initial_df['PrevWorkType'] = self.initial_df['WorkType'].shift(24)



    def create_learningSet(self, df: pd.DataFrame):
        df['DiffHistoryLoad'] = df['HistoryLoad'].diff(1)
        init_ts = df['DiffHistoryLoad']
        differences_ts_dict = {0: init_ts,
                               1: init_ts.diff(1),
                               2: init_ts.diff(2),
                               3: init_ts.diff(3),
                               4: init_ts.diff(4),
                               24: init_ts.diff(24),
                               168: init_ts.diff(168)
                               }
        lags = self.lags_dict.keys()
        for l in lags:
            diff_key = self.lags_dict.get(l)
            ts_by_diff_key = differences_ts_dict.get(diff_key)
            name = 'lag' + str(l) + 'd' + str(diff_key)
            # df[name] = ts_by_diff_key.shift(l) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WRONG
            ser = ts_by_diff_key[diff_key + 1:]
            laggs = pd.Series(np.repeat(np.nan, l))
            fin_ser = laggs.append(ser, ignore_index=True)
            fin_ser = fin_ser[:init_ts.shape[0]]
            df[name] = fin_ser

        print('count of nan = ', df.isna().sum())
        df = df.dropna()
        print('count of nan = ', df.isna().sum())
        df_to_pca = df.loc[:, 'lag5d0' : 'lag23d0']
        df = df.drop(df.columns.to_series()['lag5d0': 'lag23d0'], axis=1)
        #####################
        pca = PCA(svd_solver='randomized', n_components=10)
        # X_pca = pca.fit(df_to_pca)
        # print(pca.explained_variance_ratio_)
        # print('Sum of first ten pca = ', sum(pca.explained_variance_ratio_[0:12]))
        newData = pca.fit_transform(df_to_pca)
        names_pca = ['pca' + str(x) for x in range(10)]
        pcadf = pd.DataFrame(data=newData[0:, 0:], index=df.index, columns=names_pca)
        #####################
        df = pd.concat([df, pcadf], axis=1, sort=False, join_axes=[df.index])

        return df

    def encode_categorials_features(self, df, names_columns_list):
        df_encoded = ordinal_transform_categorials(df.copy(), names_columns_list)
        return df_encoded

    def encode_categorials_features_one_hot(self, df):
        df = one_hot_transform_categorials(df)
        return df

    def my_scale(self, df):
        df_scalled = my_min_max_scaller(df.copy())
        return df_scalled

    def scale_df(self, df: pd.DataFrame, response: str):
        # sclalle others
        df_gen_scalled = df.apply(my_min_max_scaller_series, a=1, b=3, axis=0)
        df_gen_scalled[response].iloc[-1] = np.nan
        return df_gen_scalled


    def get_train_test_set(self, df: pd.DataFrame):
        test_set = df.iloc[-1, :]
        train_set = df.iloc[:-1, :]
        return train_set, test_set


