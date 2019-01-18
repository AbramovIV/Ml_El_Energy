from NNDirectory.NNBuilderDirectory import MyCv
from NNDirectory.NNBuilderDirectory.NNParams import NNparams
from NNDirectory.PreprocessingDirectory import MyPreprocess
import pandas as pd
import numpy as np
import keras
import matplotlib.pyplot as plt


class MyTestModel:
    nn = NNparams
    cross_val = MyCv
    preprocess = MyPreprocess

    def __init__(self, nn:NNparams, cross_val: MyCv, preprocess: MyPreprocess):
        self.nn = nn
        self.cross_val = cross_val
        self.preprocess = preprocess

    def test_my_model(self, first_id, last_id):
        history_load_df = self.preprocess.learning_set[['HistoryLoad', 'Id']]

        # create log for prediction
        predicions_file_path = r'C:\Users\vgv\Desktop\PythonData\Predictions\predictions.txt'
        prediction_Headers = 'Year Month Day Time HistoryLoad Prediction Mape\n'
        f = open(predicions_file_path, "w")
        f.write(prediction_Headers)
        f.close()

        count_retrain = 0
        counter_pr = 0
        first_train = True
        trained = False
        retrain = False
        counter = 0
        for i in range(first_id, last_id):
            ls = self.preprocess.learning_set.loc[self.preprocess.learning_set.Id <= i]
            # predicted id in initial df
            prediction_row = ls.iloc[-1, :]
            id_of_predicted = ls.iloc[-1, :].Id


            prediction_history_row = self.preprocess.initital_df.loc[self.preprocess.initital_df['Id'] == id_of_predicted, ]
            prediction_ls_row = self.preprocess.learning_set.loc[self.preprocess.learning_set['Id'] == id_of_predicted, ]


            HISTORICAL_LOAD = prediction_history_row[self.preprocess.target].values[0]
            Prev_HISTORICAL_LOAD = \
                history_load_df.loc[history_load_df['Id'] == (id_of_predicted - self.preprocess.prediction_lag_diff), 'HistoryLoad'].values[0]

            # remove unness columns
            ls = ls.drop(['HistoryLoad', 'Id'], axis=1)

            current_year = ls['Year'].max()

            # prepare to neural net
            train, test = self.preprocess.get_train_test_set(ls)

            # scale train, test
            if counter != 0:
                train = self.preprocess.scale_train_df(train)
                test = self.preprocess.scale_test_df(pd.DataFrame([test.values], columns=test.index))
                x_test = test.drop(self.preprocess.response, axis=1)
                Y = train[self.preprocess.response]
                X = train.drop(self.preprocess.response, axis=1) #.values
            else:
                train = self.preprocess.scale_train_df(train)
                train['Year'] = self.preprocess.scale_collumn_prep(ls.iloc[:-1]['Year'],
                                                                   'Year',
                                                                   2017.0,
                                                                   2013.0)
                test = self.preprocess.scale_test_df(pd.DataFrame([test.values], columns=test.index))
                x_test = test.drop(self.preprocess.response, axis=1)
                Y = train[self.preprocess.response]
                X = train.drop(self.preprocess.response, axis=1)  # .values

            if count_retrain == 168 or first_train is True:

                model = self.cross_val.myCross_validation(nn=self.nn,
                                                          Xset=X,
                                                          Yset=Y,
                                                          n_folds=10,
                                                          max_epoch=1000,
                                                          trained=trained
                                                          )

                final_model = self.cross_val.train_final_model(nn=self.nn,
                                                               X=X,
                                                               Y=Y
                                                               )
                count_retrain = 0
                #self.cross_val.buildCvPlot()

            # make prediction
            prediction = final_model.predict(x_test)[0][0]

            # unscale prediction
            test[self.preprocess.response] = prediction
            unscale_prediction = self.preprocess.unsale_prediction(test)

            # undiff prediction and get final prediction value
            final_prediction = self.preprocess.undiff_pred(pred=unscale_prediction, history_lag=Prev_HISTORICAL_LOAD)

            # calc mape %
            pred_mape = self.preprocess.mape_pred(predicted=final_prediction, history=HISTORICAL_LOAD)

            print('pred_mape: ', str(pred_mape))

            # write prediction log
            prediction_log = '{0} {1} {2} {3} {4} {5} {6}\n'.format(str(prediction_history_row.iloc[0].Year), str(prediction_history_row.iloc[0].Month),
                                                                    str(prediction_history_row.iloc[0].Day), str(prediction_history_row.iloc[0].Time),
                                                                    str(HISTORICAL_LOAD), str(final_prediction),
                                                                    str(pred_mape)
                                                                    )
            with open(predicions_file_path, 'a') as f:
                f.write(prediction_log)


            count_retrain = count_retrain + 1
            counter_pr = 1 + counter_pr
            first_train=False
            trained=True
