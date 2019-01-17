import pandas as pd
from sklearn.decomposition import PCA
import numpy as np

from NNDirectory.SupportFunctions import ordinal_transform_categorials,  my_min_max_scaller_series, \
    one_hot_transform_categorials


class LearningSet:

    lags_dict = {1: 0, 2: 0, 3: 0, 4: 0, 24:0
                 #5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0, 13: 0, 14: 0,
                 #15: 0, 16: 0, 17: 0, 18: 0, 19: 0, 20: 0, 21: 0, 22: 0, 23: 0,
                 #24: 0, 48: 0, 72: 0, 96: 0, 120: 0, 144: 0, 168: 0,
                 #25: 24, 47: 24, 73: 24, 97: 24, 121: 24, 145: 24, 169: 24
                 #25: 24, 49: 24, 73: 24, 97: 24, 121: 24, 145: 24, 169: 24,
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
        #self.initial_df = self.initial_df.drop('LoadPlan', axis=1)
        self.initial_df['PrevTemp'] = self.initial_df['Temperature'].shift(24)
        self.initial_df['PrevWorkType'] = self.initial_df['WorkType'].shift(24)



    def create_learningSet(self, df: pd.DataFrame):
        df.loc[df['HistoryLoad'] <= 0, 'HistoryLoad'] = np.nan
        df['DiffHistoryLoad'] = df['HistoryLoad']#.diff(167)
        init_ts = df['DiffHistoryLoad']
        #init_ts = df['HistoryLoad']
        differences_ts_dict = {0: init_ts,
                               1: init_ts.diff(0),
                               2: init_ts.diff(0),
                               3: init_ts.diff(0),
                               4: init_ts.diff(0),
                               24: init_ts.diff(0),
                               168: init_ts.diff(0)
                               }

        lags = self.lags_dict.keys()

        for l in lags:
            diffKey = self.lags_dict.get(l)
            my_dif_ts = differences_ts_dict.get(diffKey)
            ts = my_dif_ts.shift(l)
            name = 'lag_'
            if(l >= 4 and l < 24):
                name = 'lag_pca_day'
            if(l > 24):
                 name = 'lag_pca_week'
            elif(l < 4 or l == 24):
                 name = 'lag_pca_most'
            name = name + str(l)
            df[name] = ts
        # window_4 = init_ts.shift(1).rolling(window=4)
        # means = window_4.mean()
        # window_12 = init_ts.shift(1).rolling(window=12)
        # maxs = window_12.max()
        # mins = window_12.min()
        # std = window_12.std()
        # memS = window_4.apply(self.membeshipS)
        # df['mems'] = memS
        # #df['std'] = std
        # df['means_1lag'] = means
        # df['max_1lag'] = maxs
        # df['min_1lag'] = mins
        # print('count of nan = ', df.isna().sum())


        df = df.dropna()

        # matching_day = [s for s in df.columns.values if "lag_pca_day" in s]
        # df = self.transformPca(df, matching_day, 8, 'day')
        # matching_week = [s for s in df.columns.values if "lag_pca_week" in s]
        # df = self.transformPca(df, matching_week, 5, 'week')
        # # matching_most = [s for s in df.columns.values if "lag_pca_most" in s]
        # # df = self.transformPca(df, matching_most, 3, 'most')
        return df

    def membeshipS(self, x):
        x_l = len(x)
        last = x[x_l-1]
        arr = x[:-1]
        a = min(arr)
        c= max(arr)
        b = (a+c)/2
        if last <= a:
            return 0
        if a < last and last <= b:
            return 2 * (((last - a) / (c - a)) ** 2)
        if b < last and last <= c:
            return 1 - (2 * (((last - c)/(c - a)) ** 2))
        if last > c:
            return  1
        return  x

    def transformPca(self, df, matching, n_components, namePca):
        df_to_pca = df.loc[:, matching]
        df = df.drop(df.columns.to_series()[matching], axis=1)
        #####################

        pca = PCA(svd_solver='randomized', n_components=n_components)
        # X_pca = pca.fit(df_to_pca)

        # print('Sum of first ten pca = ', sum(pca.explained_variance_ratio_[0:12]))
        ind = df_to_pca.index.values

        newData = pca.fit_transform(df_to_pca)
        print(pca.explained_variance_ratio_)
        names_pca = ['pca' + namePca + str(x) for x in range(n_components)]
        pcadf = pd.DataFrame(data=newData[0:, 0:], columns=names_pca, index=ind)
        ######################
        res = pd.concat([df, pcadf], axis=1)
        return  res




