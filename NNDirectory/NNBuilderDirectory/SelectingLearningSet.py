from random import randint

import pandas as pd
import numpy as np

class SelectingLearningSet:
    @staticmethod
    def get_ls_for_gridSearch(df:pd.DataFrame):
        ind = df.index.values
        last = ind[-1]
        train_cv = df.iloc[ : (last - 8140), :]
        valid_cv = df.iloc[(last - 8139):, :]
        return train_cv, valid_cv

    @staticmethod
    def get_ls_for_final_tain(df: pd.DataFrame):
        ind = df.index.values
        last = ind[-1]
        train_cv = df.iloc[(last - 8139): (last - 1900), ]
        valid_cv = df.iloc[last - 1899:, ]

        # create cv
        init_ind = valid_cv.index.values
        start_ind = init_ind[0]
        end_ind = init_ind[-1]
        rnd_ind = [randint(start_ind, end_ind) for p in range(start_ind, end_ind)]
        val_size = round(len(valid_cv) / 2)

        fin_val_ind = rnd_ind[:val_size]
        fin_val = valid_cv.iloc[valid_cv.index.isin(fin_val_ind)]
        ind_to_train = set(init_ind) - set(fin_val_ind)
        to_train = valid_cv.iloc[valid_cv.index.isin(ind_to_train)]
        train_cv = train_cv.append(to_train)
        return train_cv, fin_val
