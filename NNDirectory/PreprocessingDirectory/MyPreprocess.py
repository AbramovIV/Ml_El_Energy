import warnings

# import keras
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from typing import Dict, List

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, OrdinalEncoder
#
# from DataDirectory.LoadData import LoadData
# from NNDirectory.LsDirectory.LearningSetCl import LearningSet
# from NNDirectory.MyPredict import My_Predict
# from NNDirectory.NNBuilderDirectory.MyCv import MyCv
# from NNDirectory.NNBuilderDirectory.NNParams import NNparams
from sklearn.exceptions import DataConversionWarning

from NNDirectory.PreprocessingDirectory.MyMinMaxScaller import MyMinMaxScaller


class MyPreprocess:
    lags_dict_diff = Dict[int, int]
    prediction_lag_diff = int
    initital_df = pd.DataFrame
    response = str
    learning_set = pd.DataFrame
    categorial_cols = List[str]
    target = str
    num_of_categorials = int
    colls_to_one_hot = List[str]
    colls_without_onh = List[str]
    pca = PCA

    def __init__(self, initital_df: pd.DataFrame, target: str, response: str,
                 prediction_lag_diff: int, lags_dict_diff: Dict[int, int]):
        self.initital_df = initital_df
        self.prediction_lag_diff = prediction_lag_diff
        self.target = target
        self.response = response
        self.lags_dict_diff = lags_dict_diff
        self.learning_set = initital_df.copy(deep=True)
        self.num_of_categorials = 0
        warnings.filterwarnings(action='ignore', category=DataConversionWarning)


    def set_prediction_series(self):
        response_series = self.initital_df[self.target]
        response_series = response_series.diff(self.prediction_lag_diff).values
        self.learning_set[self.response] = pd.Series(response_series, index=self.learning_set.index)

    def set_prev_temp_workType(self):
        lag = [24,48,72,96,120,144,168]
        for l in lag:
            prevTemp = self.learning_set['Temperature'].shift(l).values
            name = 'temp' + str(l)
            self.learning_set[name] = pd.Series(prevTemp, index=self.learning_set.index)

        prevWorkType = self.learning_set['WorkType'].shift(24).values
        self.learning_set['PrevWorkType'] = pd.Series(prevWorkType, index=self.learning_set.index)





    def set_augmentations_lags(self, isDiff:bool):
        init_series = self.learning_set[self.response]
        if isDiff is False:
            for lag in self.lags_dict_diff.keys():
                lag_series = init_series.shift(lag).values
                name = 'lag' + str(lag)
                self.learning_set[name] = pd.Series(lag_series, index=self.learning_set.index)
        else:
            for lag in self.lags_dict_diff.keys():
                diff_key = self.lags_dict_diff.get(lag)
                if diff_key == 0:
                    lag_series = init_series.shift(lag)
                    name = 'lag' + str(lag)
                    self.learning_set[name] = pd.Series(lag_series, index=self.learning_set.index)
                    continue
                else:
                    diff_series = init_series.diff(diff_key)
                    lag_series = diff_series.shift(lag).values
                    name = 'lag' + str(lag)
                    self.learning_set[name] = pd.Series(lag_series, index=self.learning_set.index)

    def init_pca(self, df, matching, n_components):
        df_to_pca = df.loc[:, matching]
        self.pca = PCA(svd_solver='randomized', n_components=n_components)
        self.pca.fit(df_to_pca)
        print('Sum of pca = ', sum(self.pca.explained_variance_ratio_[0:n_components]))

    def transformPca(self, df, matching, n_components):
        df_to_pca = df.loc[:, matching]
        df = df.drop(df.columns.to_series()[matching], axis=1)
        ind = df_to_pca.index.values
        newData = self.pca.transform(df_to_pca)
        names_pca = ['pca' + str(x) for x in range(n_components)]
        pcadf = pd.DataFrame(data=newData[0:, 0:], columns=names_pca, index=ind)
        res = pd.concat([df, pcadf], axis=1)
        return res

    def encode_one_hot(self, df: pd.DataFrame):
        df_c = df.copy(deep=True)

        for onh in self.colls_to_one_hot:
            dummies = pd.get_dummies(df_c[onh], prefix=onh, drop_first=False)
            dummies = dummies.replace(1, 0.5)
            df_c = pd.concat([df_c, dummies], axis=1)
            self.num_of_categorials = self.num_of_categorials + dummies.shape[1]
        df_c = df_c.drop(self.colls_to_one_hot, axis=1)
        return df_c

    def scale_train_df(self, df: pd.DataFrame):
        self.colls_without_onh = set([s for s in df.columns.values if not "_" in s])
        self.colls_without_onh = self.colls_without_onh.difference(set(['HistoryLoad', 'Id']))
        #df_c = df.copy(deep=True)
        df_c = df
        old_indexis = df_c.index.values
        #self.min_max_scaler = MinMaxScaler(feature_range=(0, 1))
        self.min_max_scaler = MyMinMaxScaller(low_range=-0.5, top_range=0.5, df_colls=self.colls_without_onh)

        x_scaled = self.min_max_scaler.transform(df_c)
        # x_scaled = self.fix_categorials(x_scaled)
        #res = pd.DataFrame(x_scaled, columns=df_c.columns.values, index=old_indexis)
        return x_scaled

    def scale_test_df(self, df: pd.DataFrame):
        #df_c = df.copy(deep=True)
        df_c = df
        old_indexis = df_c.index.values
        x_scaled = self.min_max_scaler.scale(df_c)
        #x_scaled = self.fix_categorials(x_scaled)
        #res = pd.DataFrame(x_scaled, columns=df_c.columns.values, index=old_indexis)
        #res.iloc[0][self.response] = np.nan
        return x_scaled

    def scale_collumn_prep(self, ser: pd.Series, coll: str, max_val, min_val):
        return self.min_max_scaler.scale_collumn(ser, coll, max_val, min_val)

    def unsale_prediction(self, df: pd.DataFrame):
        unscale = self.min_max_scaler.inverse_transform(df)
        #res_df = pd.DataFrame(data=unscale, columns=df.columns.values)
        #res = res_df.iloc[0][self.response]
        return unscale.iloc[0][self.response]

    def get_train_test_set(self, df: pd.DataFrame):
        test_set = df.iloc[-1, :]
        train_set = df.iloc[:-1, :]
        return train_set, test_set

    def undiff_pred(self, pred, history_lag):
        return (history_lag + pred)

    def mape_pred(self, predicted, history):
        return abs((history - predicted) / history) * 100.0

    # def fix_categorials(self, df: pd.DataFrame):
    #     #df_c = df.copy(deep=True)
    #     df_c = df
    #     matching_colls = [s for s in df.columns.values if "_" in s]
    #     for coll in matching_colls:
    #         df_c[coll] = df_c[coll].replace(-1, 0)
    #     return df_c

    def init_label_encoder(self, df: pd.DataFrame):
        df_c = df
        df_label = df_c.loc[:, self.categorial_cols]
        self.labelenc = OrdinalEncoder()
        self.labelenc.fit(df_label)

    def label_encode(self, df: pd.DataFrame):
        # df_c = df.copy(deep=True)
        df_c = df
        colls = df_c.columns.values
        without_label = set(colls) - set(self.categorial_cols)
        df_label = df_c.loc[:, self.categorial_cols]
        df_without_label = df_c.drop(self.categorial_cols, axis=1)
        labeled = self.labelenc.transform(df_label)
        labeled_df = pd.DataFrame(labeled, columns=df_label.columns.values, index=df_label.index)
        res = pd.concat([df_without_label, labeled_df], axis=1, sort=False, join_axes=[df_c.index])
        return res












