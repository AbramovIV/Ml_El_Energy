from NNDirectory.LearningSetCl import LearningSet
from NNDirectory.NNBuilderDirectory.MyCv import MyCv
from NNDirectory.NNBuilderDirectory.NNParams import NNparams
import keras
import numpy as np

from NNDirectory.SupportFunctions import unscale_el_load_pred, undiffPred, mape_pred


def create_train_test_ls(ls):
    # drop ids column
    ls = ls.drop('Id', axis=1)
    # AllColumnsCategorial
    categorial_cols = ['Year', 'Day', 'DayName', 'WorkType', 'PrevWorkType', 'Time']

    # encode_one_hot
    one_hot_columns = ['Time', 'DayName']  # , 'Month', 'DayName', 'WorkType', 'PrevWorkType'
    df_to_one_hot = ls.loc[:, one_hot_columns]
    one_hot_df = ls_obj.encode_categorials_features_one_hot(df_to_one_hot)

    # encode ordinal
    columns_to_ordinal = set(categorial_cols) - set(one_hot_columns)
    columns_without_one_hot = ls.columns.difference(one_hot_columns)

    df_to_ordinal = ls.loc[:, columns_without_one_hot]
    ordinal_encoded = ls_obj.encode_categorials_features(df_to_ordinal, columns_to_ordinal)

    scalled_ordinal_encoded = ls_obj.scale_df(df=ordinal_encoded, response=response)

    final_learning_set = scalled_ordinal_encoded.join(one_hot_df)

    # prepare to neural net
    train, test = ls_obj.get_train_test_set(final_learning_set)
    return train, test

ls_obj = LearningSet(path_to_df=r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt')
response = 'DiffHistoryLoad'

# HISTORICAL VALUES OF ELECTRICITY LOAD
history_load_df = ls_obj.initial_df[['HistoryLoad', 'Id']]

my_df = ls_obj.create_learningSet(ls_obj.initial_df)
my_df = my_df.drop(['HistoryLoad'], axis=1)
my_df = my_df.reset_index(drop=True)


first_pred_ind = my_df.index[my_df['Year'] == 2017][0]
first_pred_ind = first_pred_ind + 1
last_pred_ind = my_df.index[-1]
ls = my_df.iloc[:first_pred_ind, :]

# predicted id in initial df
id_of_predicted = ls.iloc[-1, :].Id

HISTORICAL_LOAD = history_load_df.loc[history_load_df['Id'] == id_of_predicted, 'HistoryLoad'].values[0]
Prev_HISTORICAL_LOAD = history_load_df.loc[history_load_df['Id'] == (id_of_predicted - 1), 'HistoryLoad'].values[0]

# create train and test sets
train, test = create_train_test_ls(ls.copy())

Y = train[response].values
X = train.drop(response, axis=1).values
x_test = test.drop(response, axis=0).values

# create nn
input_dim = X.shape[1]
batch_size = 168

#create nn model with parameters
nn = NNparams(hidden=[25], dropout=[0.0, 0.0],
              optimizer=keras.optimizers.Adam(amsgrad=True), l1reg=0, l2reg=0,
              maxEpoch_gridsearch=500000, activation='relu', input_dim=input_dim,
              loss='mean_squared_error', batch_size=batch_size)

nn_model = nn.buildNNModel()

# create cross-validation
mycv = MyCv(model_filepath=r"C:\Users\vgv\Desktop\PythonData\Predictions\weights.best.hdf5")

# define parameters for early stop FOR FIRST TRAIN NN
early_stop_first_train = keras.callbacks.EarlyStopping(monitor='mean_absolute_error',
                                                       min_delta=1e-4,
                                                       patience=10,
                                                       verbose=2,
                                                       mode='auto'
                                                       )
# model training and cv
model = mycv.myCross_validation(nn_model, X=X, Y=Y, n_folds=10, max_epoch=nn.maxEpoch_gridsearch, batch_size=batch_size,
                                early_stop=early_stop_first_train)

# final model training
final_model = mycv.final_model(nn_model=model, X=X, Y=Y, batch_size=batch_size, early_stop=early_stop_first_train)

# mycv.buildCvPlot()

early_stop = keras.callbacks.EarlyStopping(monitor='loss',
                                           min_delta=1e-4,
                                           patience=5,
                                           verbose=2,
                                           mode='auto'
                                           )
# create log file for predictions
predicions_file_path = r'C:\Users\vgv\Desktop\PythonData\Predictions\predictions.txt'
prediction_Headers = 'Year Month Day Time HistoryLoad Prediction Mape\n'
f = open(predicions_file_path, "w")
f.write(prediction_Headers)
f.close()
count_retrain = 0

def test_model(my_df, first_pred_ind, last_pred_ind):
    for i in range(first_pred_ind, last_pred_ind):
        ls = my_df.iloc[:i, :]

        # predicted id in initial df
        prediction_row = ls.iloc[-1, :]
        id_of_predicted = ls.iloc[-1, :].Id
        HISTORICAL_LOAD = history_load_df.loc[history_load_df['Id'] == id_of_predicted, 'HistoryLoad'].values[0]
        Prev_HISTORICAL_LOAD = history_load_df.loc[history_load_df['Id'] == (id_of_predicted - 1), 'HistoryLoad'].values[0]


        # prepare to neural net
        train, test = create_train_test_ls(ls)
        Y = train[response].values
        X = train.drop(response, axis=1).values
        x_test = test.drop(response, axis=0).values
        if count_retrain == 24:

            model = mycv.myCross_validation(nn_model, X=X, Y=Y, n_folds=5, max_epoch=nn.maxEpoch_gridsearch,
                                            batch_size=batch_size,
                                            early_stop=early_stop)
            final_model = mycv.final_model(nn_model=model, X=X, Y=Y, batch_size=batch_size, early_stop=early_stop)
            count_retrain = 0

        prediction = final_model.predict(np.array([x_test]))[0][0]

        unscale_prediction = unscale_el_load_pred(pred=prediction, a=1, b=3, el_train=ls.ix[-1, response])
        final_prediction = undiffPred(pred=unscale_prediction, history_lag=Prev_HISTORICAL_LOAD)
        pred_mape = mape_pred(predicted=final_prediction, history=HISTORICAL_LOAD)

        prediction_log = '{0} {1} {2} {3} {4} {5} {6}\n'.format(str(prediction_row.Year), str(prediction_row.Month),
                                                                str(prediction_row.Day), str(prediction_row.Time),
                                                                str(HISTORICAL_LOAD), str(final_prediction), str(pred_mape)
                                                                )
        with open(predicions_file_path, 'a') as f:
            f.write(prediction_log)

        print('pred_mape: ', pred_mape)
        count_retrain = count_retrain + 1
