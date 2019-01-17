from NNDirectory.LsDirectory.Prepare_Ls import Prepare_Ls
from NNDirectory.NNBuilderDirectory.MyCv import MyCv
from NNDirectory.NNBuilderDirectory.NNParams import NNparams
import keras
import numpy as np
import pandas as pd

from NNDirectory.SupportFunctions import unscale_el_load_pred, undiffPred, mape_pred


class My_Predict:
    nn = NNparams
    my_df = pd.DataFrame
    response = str
    cross_val = MyCv
    prepare_ls = Prepare_Ls

    def __init__(self, my_df: pd.DataFrame, nn: NNparams, response: str, cross_val: MyCv, prepare_ls: Prepare_Ls):
        self.nn = nn
        self.my_df = my_df
        self.response = response
        self.cross_val = cross_val
        self.prepare_ls = prepare_ls

    def test_my_model(self, first_id, last_id):
        history_load_df = self.my_df[['HistoryLoad', 'Id']]

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
        self.nn.get_nn_model().save_weights(self.cross_val.model_initial_weights)
        #a = self.my_df.iloc[first_pred_ind - 9000 : ,]
        #import matplotlib.pyplot as plt
        #plt.plot(a.DiffHistoryLoad.values)
        #plt.show()


        for i in range(first_id, last_id):
            ls = self.my_df.loc[self.my_df.Id <= i]
            # predicted id in initial df
            prediction_row = ls.iloc[-1, :]
            id_of_predicted = ls.iloc[-1, :].Id
            HISTORICAL_LOAD = history_load_df.loc[history_load_df['Id'] == id_of_predicted, 'HistoryLoad'].values[0]
            Prev_HISTORICAL_LOAD = \
                history_load_df.loc[history_load_df['Id'] == (id_of_predicted - 168), 'HistoryLoad'].values[0]

            # remove unness columns
            ls = ls.drop(['HistoryLoad', 'Id'], axis=1)
            # prepare final learning set
            ls_encoded = self.prepare_ls.encode_and_scale_ls_factors(ls)
            # prepare to neural net
            train, test = self.prepare_ls.get_train_test_set(ls_encoded)
            Y = train[self.response]
            X = train.drop(self.response, axis=1)#.values
            x_test = test.drop(self.response, axis=0)#.values

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

            prediction = final_model.predict(np.array([x_test]))[0][0]

            # unscale_prediction = unscale_el_load_pred(pred=prediction, a=1, b=3, el_train=ls[self.response].iloc[:-1])
            test['DiffHistoryLoad'] = prediction
            unscale_prediction = self.prepare_ls.unscale_prediction(test)
            #final_prediction = undiffPred(pred=unscale_prediction, history_lag=Prev_HISTORICAL_LOAD)
            final_prediction = unscale_prediction
            pred_mape = mape_pred(predicted=final_prediction, history=HISTORICAL_LOAD)

            prediction_log = '{0} {1} {2} {3} {4} {5} {6}\n'.format(str(prediction_row.Year), str(prediction_row.Month),
                                                                    str(prediction_row.Day), str(prediction_row.Time),
                                                                    str(HISTORICAL_LOAD), str(final_prediction),
                                                                    str(pred_mape)
                                                                    )
            with open(predicions_file_path, 'a') as f:
                f.write(prediction_log)

            print('pred_mape: ', pred_mape)
            count_retrain = count_retrain + 1
            counter_pr = 1 + counter_pr
            first_train=False
            trained=True
