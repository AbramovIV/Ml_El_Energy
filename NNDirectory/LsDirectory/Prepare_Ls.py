import string
from typing import List
import pandas as pd
import numpy as np
from NNDirectory.SupportFunctions import one_hot_transform_categorials, ordinal_transform_categorials, \
    my_min_max_scaller, my_min_max_scaller_series


class Prepare_Ls:
    categorial_cols = List[str]
    ordinal_encoding_names = List[str]
    one_hot_encoding_names = List[str]
    predictors = List[str]
    response = str

    def __init__(self, categorial_cols, one_hot_encoding_names, response):
        self.categorial_cols = categorial_cols
        self.one_hot_encoding_names = one_hot_encoding_names
        self.ordinal_encoding_names = set(categorial_cols) - set(one_hot_encoding_names)
        self.response = response

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

    def encode_and_scale_ls_factors(self, df: pd.DataFrame):
        if len(self.one_hot_encoding_names) > 0:
            #encode one hot
            df_to_one_hot = df.loc[:, self.one_hot_encoding_names]
            one_hot_df = self.encode_categorials_features_one_hot(df_to_one_hot)

            # encode ordinal
            columns_to_ordinal = set(self.categorial_cols) - set(self.one_hot_encoding_names)
            columns_without_one_hot = df.columns.difference(self.one_hot_encoding_names)
            df_to_ordinal = df.loc[:, columns_without_one_hot]
            ordinal_encoded = self.encode_categorials_features(df_to_ordinal, columns_to_ordinal)
            # scale
            scalled_ordinal_encoded = self.scale_df(df=ordinal_encoded, response=self.response)
            # join ordinal and one hot encoding df
            final_learning_set = scalled_ordinal_encoded.join(one_hot_df)
        else:
            df_to_ordinal = df
            ordinal_encoded = self.encode_categorials_features(df_to_ordinal, self.categorial_cols)
            scalled_ordinal_encoded = self.scale_df(df=ordinal_encoded, response=self.response)
            final_learning_set = scalled_ordinal_encoded

        return final_learning_set

    def get_nums_of_predictors(self, df: pd.DataFrame):
        return self.encode_and_scale_ls_factors(df).shape[1]
