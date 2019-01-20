import numpy as np
import pandas as pd
from typing import Dict, List


class MyMinMaxScaller:
    a = int
    b = int
    max_dict = None
    min_dict = None
    df_colls = List[str]

    def __init__(self, low_range: int, top_range: int, df_colls):
        self.a = low_range
        self.b = top_range
        self.df_colls = df_colls

    def transform(self, df: pd.DataFrame):
        self.max_dict = {}
        self.min_dict = {}

        # df_c = df.copy(deep=True)
        df_c = df
        df_colls = df_c.columns.values

        for coll in self.df_colls:
            series = df_c[coll]
            val_max = series.max()
            val_min = series.min()
            self.max_dict[coll] = val_max
            self.min_dict[coll] = val_min
            df_c[coll] = self.my_min_max_scaller_series(df_c[coll], val_max, val_min)

        return df_c

    def scale(self, df: pd.DataFrame):
        df_c = df.copy(deep=True)
        df_colls = df_c.columns.values

        for coll in self.df_colls:
            series = df_c[coll]
            val_max = series.max()
            val_min = series.min()
            val_max = self.max_dict.get(coll)
            val_min = self.min_dict.get(coll)
            self.min_dict[coll] = val_min
            df_c[coll] = self.my_min_max_scaller_series(df_c[coll], val_max, val_min)

        return df_c

    def scale_collumn(self, ser: pd.Series, coll: str, max_val, min_val):
        self.max_dict.update({coll: max_val})
        self.min_dict.update({coll: min_val})
        return self.my_min_max_scaller_series(ser, max_val, min_val)

    def my_min_max_scaller_series(self, x: pd.Series, x_max, x_min):
        return (self.b - self.a) * ((x - x_min) / (x_max - x_min)) + self.a

    def my_min_max_unscaller_series(self, x: pd.Series, coll:str, x_max, x_min):
        x_max = self.max_dict.get(coll)
        x_min = self.min_dict.get(coll)
        return (((x - self.a) * (x_max - x_min)) / (self.b - self.a)) + x_min

    def inverse_transform(self, df: pd.DataFrame):
        df_c = df.copy(deep=True)
        df_colls = df_c.columns.values

        for coll in self.df_colls:
            series = df_c[coll]
            val_max = series.max()
            val_min = series.min()
            x_max = self.max_dict.get(coll)
            x_min = self.min_dict.get(coll)
            df_c[coll] = self.my_min_max_unscaller_series(df_c[coll], coll, x_max, x_min)

        return df_c