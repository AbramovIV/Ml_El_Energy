from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h2o
from h2o.estimators import H2ODeepLearningEstimator
from h2o.grid.grid_search import H2OGridSearch

from NNDirectory.LsDirectory import Prepare_Ls
from NNDirectory.NNBuilderDirectory.SelectingLearningSet import SelectingLearningSet
from NNDirectory.PreprocessingDirectory.MyPreprocess import MyPreprocess


class NNH2o:
    input_neurons = int
    hyper_params = Dict
    response = str
    predictors = List[str]
    ls = pd.DataFrame
    training_fr_grid_search = pd.DataFrame
    valid_fr_grid_search = pd.DataFrame
    listNNModels = List
    listCheckpointsNN = List
    preprocess = MyPreprocess
    df_init = pd.DataFrame
    mathing_pca = List[str]
    prepare_ls = Prepare_Ls
    ncomponents = int

    def __init__(self, input_neurons: int, response: str, predictors: List[str], df_init: pd.DataFrame,
                 preprocess:MyPreprocess,
                 prepare_ls: Prepare_Ls):
        h2o.init()
        self.input_neurons = input_neurons
        self.hyper_params = self.define_hyper_params()
        self.response = response
        self.predictors = predictors
        self.df_init = df_init

        self.preprocess = preprocess
        self.mathing_pca = [s for s in self.df_init.columns.values if "lag" in s]
        self.prepare_ls = prepare_ls
        self.ncomponents = 15
        self.badflag = True

    def get_perc(self, per, num):
        return round(abs(num * per) / 100)

    def create_ls(self, df: pd.DataFrame):
        self.preprocess.init_pca(df, self.mathing_pca, self.ncomponents)
        ls = self.preprocess.transformPca(df, self.mathing_pca, self.ncomponents)
        ls = self.preprocess.scale_train_df(ls)
        if(self.badflag is True):
            self.predictors = list(ls)
            self.predictors.remove(self.response)
            self.predictors.remove('Id')
            self.predictors.remove('HistoryLoad')
            self.badflag = False

        return ls


    def define_hyper_params(self):
        hyper_params = {
            "activation": ["Tanh"],
            "hidden": [
                [self.input_neurons],
                [self.input_neurons, self.get_perc(50, self.input_neurons)],
                [self.input_neurons, self.get_perc(50, self.input_neurons), self.get_perc(20, self.input_neurons)],
                [self.input_neurons*2]
            ],
            'rate': [0.001, 0.005, 1e-4],
            'rate_annealing': [1e-8, 1e-7, 1e-6],
            "distribution": ["laplace"]
        }
        return hyper_params

    def create_train_valid_samples_gridsearch(self, df: pd.DataFrame):
        ind = df.index.values
        last = ind[-1]
        self.training_fr_grid_search = df.iloc[: (last - 8140), :]
        self.valid_fr_grid_search = df.iloc[(last - 8139):, :]
        self.training_fr_grid_search = h2o.H2OFrame(self.training_fr_grid_search)
        self.valid_fr_grid_search = h2o.H2OFrame(self.valid_fr_grid_search)

        # self.training_fr_grid_search["Month"] = self.training_fr_grid_search["Month"].asfactor()
        # self.training_fr_grid_search["DayName"] = self.training_fr_grid_search["DayName"].asfactor()
        # self.training_fr_grid_search["WorkType"] = self.training_fr_grid_search["WorkType"].asfactor()
        # self.training_fr_grid_search["Time"] = self.training_fr_grid_search["Time"].asfactor()
        #
        # self.valid_fr_grid_search["Month"] = self.valid_fr_grid_search["Month"].asfactor()
        # self.valid_fr_grid_search["DayName"] = self.valid_fr_grid_search["DayName"].asfactor()
        # self.valid_fr_grid_search["WorkType"] = self.valid_fr_grid_search["WorkType"].asfactor()
        # self.valid_fr_grid_search["Time"] = self.valid_fr_grid_search["Time"].asfactor()

    def run_randomgrid(self):
        search_criteria = {"strategy": "RandomDiscrete", "max_models": 10, "max_runtime_secs": 30, "seed": 123456}
        model_grid = H2OGridSearch(H2ODeepLearningEstimator, hyper_params=self.hyper_params, search_criteria=search_criteria)

        model_grid.train(
            # algorithm="deeplearning",
            grid_id="dl_grid_random",
            training_frame=self.training_fr_grid_search,
            validation_frame=self.valid_fr_grid_search,
            x=self.predictors,
            y=self.response,
            categorical_encoding="auto",
            standardize=False,
            epochs=10,
            adaptive_rate=False,
            nesterov_accelerated_gradient=True,
            # rate=1e-3,
            shuffle_training_data=False,
            stopping_metric="mae",
            stopping_tolerance=1e-2,  ## stop when mae does not improve by >=1% for 2 scoring events
            stopping_rounds=5,
            mini_batch_size=168,
            train_samples_per_iteration=0,
            score_validation_samples=0,  ## downsample validation set for faster scoring#
            score_duty_cycle=0.025,  ## don't score more than 2.5% of the wall time
            max_w2=10  ## can help improve stability for Rectifier
        )
        sorted_model_grid = model_grid.get_grid(sort_by='mae', decreasing=False)
        print(sorted_model_grid)

        first_model = sorted_model_grid[0]  ## model with lowest mae
        second_model = sorted_model_grid[1]
        third_model = sorted_model_grid[2]

        self.listNNModels = list()
        self.listNNModels.append(first_model)
        self.listNNModels.append(second_model)
        self.listNNModels.append(third_model)
        NNH2o.print_model_params(self.listNNModels, False)

    def final_train(self, train: pd.DataFrame, valid: pd.DataFrame):
        train_hex = h2o.H2OFrame(train)
        valid_hex = h2o.H2OFrame(valid)
        self.listCheckpointsNN = list()
        counter = 1
        for model in self.listNNModels:
            id = model.model_id
            name = str(model.model_id) + str(counter)
            model_chkp = H2ODeepLearningEstimator(
                checkpoint=model.model_id,
                model_id=name,
                activation=model.actual_params.get("acivation"),
                training_frame=train_hex,
                validation_frame=valid_hex,
                # x=self.predictors,
                # y=self.response,
                stopping_tolerance=1e-4,
                stopping_rounds=3,
                mini_batch_size=24,
                epochs=1e6,
                hidden=model.actual_params.get("hidden"),
                rate=model.actual_params.get("rate"),
                rate_annealing= model.actual_params.get("rate_annealing"),
                distribution= model.actual_params.get("distribution"),
                categorical_encoding=model.actual_params.get("categorical_encoding"),
                standardize=model.actual_params.get("standardize"),
                adaptive_rate=model.actual_params.get("adaptive_rate"),
                nesterov_accelerated_gradient=model.actual_params.get("nesterov_accelerated_gradient"),
                shuffle_training_data=model.actual_params.get("shuffle_training_data"),
                stopping_metric=model.actual_params.get("stopping_metric"),
                train_samples_per_iteration=0,
                score_validation_samples=0,  ## downsample validation set for faster scoring#
                score_duty_cycle=0.025,  ## don't score more than 2.5% of the wall time
                max_w2=model.actual_params.get("max_w2")  ## can help improve stability for Rectifier
            )
            model_chkp.train(
                x=self.predictors,
                y=self.response,
                training_frame=train_hex,
                validation_frame=valid_hex
            )
            counter = counter + 1
            self.listCheckpointsNN.append(model_chkp)
            #self.listCheckpointsNN.append(model)
        self.listNNModels = self.listCheckpointsNN
        NNH2o.print_model_params(self.listNNModels, False)

    @staticmethod
    def print_model_params(listNNModels, plotting: bool):
            for model in listNNModels:
                print(model.model_id + " activation: " + model.actual_params.get('activation') +
                      " hidden " + ','.join([str(elem) for elem in model.actual_params.get('hidden')]) +
                      " l1 " + str(model.actual_params.get('l1')) +
                      " l2 " + str(model.actual_params.get('l2')) +
                      " mae: " + str(model.mae(valid=True)) +
                      " rate: " + str(model.actual_params.get('rate')) +
                      " rate_annealing: " + str(model.actual_params.get('rate_annealing')) +
                      " distribution " + model.actual_params.get('distribution'))
                sh = model.score_history()
                sh = pd.DataFrame(sh)
                print(sh)
                if plotting is True:
                    sh.plot(x='epochs', y=['training_mae', 'validation_mae'])
                    plt.show()

    def create_file_log(self):
        # create log for prediction
        predicions_file_path = r'C:\Users\vgv\Desktop\PythonData\Predictions\predictions.txt'
        prediction_Headers = 'Year Month Day WorkType Time HistoryLoad Prediction Mape\n'
        f = open(predicions_file_path, "w")
        f.write(prediction_Headers)
        f.close()

    def learn_first_models(self, df: pd.DataFrame):
        # create ls
        df = self.df_init.loc[self.df_init['Year'] <= 2016]

        ls = self.create_ls(df=df)
        # remove unness columns
        ls = ls.drop(['HistoryLoad', 'Id'], axis=1)
        self.create_train_valid_samples_gridsearch(ls)
        # create and select models
        self.run_randomgrid()

        # checkpointing training
        train_final, valid_final = SelectingLearningSet.get_ls_for_final_tain(ls)
        self.final_train(train=train_final, valid=valid_final)

    def make_predict(self, test_hex):
        prediction_list = list()
        for model in self.listNNModels:
            pred = model.predict(test_hex)
            prediction_list.append(pred)
        return  prediction_list

    def unscale_predictions(self, list_pred):
        unscale_list = list()
        for pred in list_pred:
            unsc = self.prepare_ls.unscale_prediction(pred)
            unscale_list.append(unsc)
        return unscale_list

    def undiff_pred(self, prevDiffLoad, list_of_unscale_pred):
        final_pred = list()
        for pred in list_of_unscale_pred:
            fin = prevDiffLoad + pred
            final_pred.append(fin)
        return final_pred

    def run_test(self, first_id: int, last_id: int):
        # create models
        self.learn_first_models(self.df_init)

        history_load_df = self.df_init[['HistoryLoad', 'Id']]
        self.create_file_log()
        iter_accum = 1

        for i in range(first_id, last_id):
            df = self.df_init.loc[self.df_init.Id <= i]

            # retrain model
            if iter_accum == 168:
                # rebuild pca components
                train = self.create_ls(df.iloc[:-1, :])
                self.final_train(train=train)
                iter_accum = 1

            # create ls with old pca
            ls = self.preprocess.transformPca(df, self.mathing_pca, self.ncomponents)
            ls = self.preprocess.scale_train_df(ls.learning_set)

            # find history values
            prediction_row = ls.iloc[-1, :]
            id_of_predicted = ls.iloc[-1, :].Id
            HISTORICAL_LOAD = history_load_df.loc[history_load_df['Id'] == id_of_predicted, 'HistoryLoad'].values[0]
            Prev_HISTORICAL_LOAD = history_load_df.loc[history_load_df['Id'] == (id_of_predicted - 1), 'HistoryLoad'].values[0]

            # remove unness columns
            ls = ls.drop(['HistoryLoad', 'Id'], axis=1)

            test = ls.iloc[:-1, :]
            test.hex = h2o.H2OFrame(test)

            predictions = self.make_predict(test.hex)
            unsc_pred = self.unscale_predictions(predictions)
            final_pred = self.undiff_pred()


            iter_accum = iter_accum + 1





