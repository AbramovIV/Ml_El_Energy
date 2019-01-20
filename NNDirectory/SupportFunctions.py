import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder


def ordinal_transform_categorials(df, col_names):
    cl_enc = list(set(df.columns.values) - set(col_names))
    df_to_enc = df.drop(cl_enc, axis=1)
    enc_col_names = df_to_enc.columns.values
    df_not_enc = df.drop(col_names, axis=1)
    enc = LabelEncoder()
    df_to_enc_arr = enc.fit_transform(df_to_enc)
    df_to_enc = pd.DataFrame(data=df_to_enc_arr[0:, 0:], index=df_to_enc.index, columns=enc_col_names)
    df = pd.concat([df_to_enc, df_not_enc], axis=1, sort=False, join_axes=[df.index])
    return df

def one_hot_transform_categorials(df):
    col_names = df.columns
    for i in col_names:
        one_hot = pd.get_dummies(df[i], prefix=i, drop_first=False)
        df = df.drop(i, axis=1)
        df = df.join(one_hot)
    return df


def my_min_max_scaller(df: pd.DataFrame):
    scaler = MinMaxScaler(feature_range=(-1, 1), copy=True)
    res_scalled = scaler.fit_transform(df)
    res_scalled_df = pd.DataFrame(data=res_scalled[0:, 0:], index=df.index, columns=df.columns.values)
    return res_scalled_df

def my_min_max_scaller_series(x: pd.Series, a, b):
    return (b - a) * ((x - x.min()) / (x.max() - x.min())) - a

def unscale_el_load_pred(pred, a, b, el_train):
    valMax = el_train.max()
    valMin = el_train.min()
    return (((pred + a) * (valMax - valMin)) / (b - a)) + valMin

def undiffPred(pred, history_lag):
    return (history_lag + pred)

def mape_pred(predicted, history):
    return abs((history - predicted)/history)*100.0
