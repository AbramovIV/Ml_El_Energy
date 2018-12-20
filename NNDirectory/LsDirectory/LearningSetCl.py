import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from NNDirectory.SupportFunctions import ordinal_transform_categorials, my_min_max_scaller, my_min_max_scaller_series, \
    one_hot_transform_categorials


class LearningSet:

    lags_dict = {1: 0, 2: 1, 3: 2, 4: 3,
                 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                 15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
                 24: 24, 48: 24, 72: 24, 96: 24, 120: 24, 144: 24, 168: 168
                }
    #lags_dict = {1: 0, 2: 0, 3: 0, 4: 0,
    #             5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
    #             15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
    #             24: 0, 48: 0, 72: 0, 96: 0, 120: 0, 144: 0, 168: 0
    #             }
    #lags_dict = {1: 0, 2: 24, 3: 24, 4: 24, 24: 24, 48: 24, 72: 24, 96: 24, 120: 24, 144: 24, 168: 168
    #             }
#
    initial_df = None
    learning_set = None

    def __init__(self, path_to_df: str) -> None:
        self.initial_df = pd.read_table(path_to_df, sep=" ", header=0, na_values='nan', keep_default_na=True)
        self.initial_df['PrevTemp'] = self.initial_df['Temperature'].shift(24)
        self.initial_df['PrevWorkType'] = self.initial_df['WorkType'].shift(24)



    def create_learningSet(self, df: pd.DataFrame):
        df['DiffHistoryLoad'] = df['HistoryLoad'].diff(1)
        init_ts = df['DiffHistoryLoad']
        #init_ts = df['HistoryLoad']
        differences_ts_dict = {0: init_ts,
                               1: init_ts.diff(1),
                               2: init_ts.diff(2),
                               3: init_ts.diff(3),
                               4: init_ts.diff(4),
                               24: init_ts.diff(24),
                               168: init_ts.diff(168)
                               }

        lags = self.lags_dict.keys()
        #for l in lags:
        #    diff_key = self.lags_dict.get(l)
        #    ts_by_diff_key = differences_ts_dict.get(diff_key)
        #    name = 'lag' + str(l) + 'd' + str(diff_key)
        #    # df[name] = ts_by_diff_key.shift(l) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WRONG
        #    ser = ts_by_diff_key[diff_key + 1:]
        #    laggs = pd.Series(np.repeat(np.nan, l+1))
        #    fin_ser = laggs.append(ser, ignore_index=True)
        #    fin_ser = fin_ser[:init_ts.shape[0]]
        #    df[name] = fin_ser
        for l in lags:
            diff_key = self.lags_dict.get(l)
            ts_by_diff_key = differences_ts_dict.get(diff_key)
            name = 'lag' + str(l) + 'd' + str(diff_key)
            # df[name] = ts_by_diff_key.shift(l) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!WRONG
            fin_ser = ts_by_diff_key.shift(l)
            df[name] = fin_ser
        window = df['lag1d0'].rolling(window=4)
        means = window.mean()
        maxs = window.max()
        mins = window.min()
        df['means_1lag'] = means
        df['max_1lag'] = maxs
        df['min_1lag'] = mins
        print('count of nan = ', df.isna().sum())

        df = df.dropna()
        print('count of nan = ', df.isna().sum())
        df_to_pca = df.loc[:, 'lag5d0' : 'lag23d0']
        df = df.drop(df.columns.to_series()['lag5d0': 'lag23d0'], axis=1)
        #####################
        n_components = 9
        pca = PCA(svd_solver='randomized', n_components=n_components)
        # X_pca = pca.fit(df_to_pca)
        # print(pca.explained_variance_ratio_)
        # print('Sum of first ten pca = ', sum(pca.explained_variance_ratio_[0:12]))
        newData = pca.fit_transform(df_to_pca)
        names_pca = ['pca' + str(x) for x in range(n_components)]
        pcadf = pd.DataFrame(data=newData[0:, 0:], index=df.index, columns=names_pca)
        #####################
        df = pd.concat([df, pcadf], axis=1, sort=False, join_axes=[df.index])
#
        return df




