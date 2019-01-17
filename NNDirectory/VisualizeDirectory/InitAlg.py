import keras
from NNDirectory.LsDirectory.LearningSetCl import LearningSet
from NNDirectory.LsDirectory.Prepare_Ls import Prepare_Ls
from NNDirectory.MyPredict import My_Predict
from NNDirectory.NNBuilderDirectory.MyCv import MyCv
from NNDirectory.NNBuilderDirectory.NNParams import NNparams
import matplotlib.pyplot as plt
from random import randint

# load data



ls_obj = LearningSet(path_to_df=r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt')
response = 'DiffHistoryLoad'
my_df = ls_obj.create_learningSet(ls_obj.initial_df)

#my_df = my_df.loc[my_df['Year'] > 2014, :]


my_df = my_df.reset_index(drop=True)  # CHECK THIS


my_df.loc[my_df['DayName'] == 'Mon', 'DayName'] = 1
my_df.loc[my_df['DayName'] == 'Tue', 'DayName'] = 2
my_df.loc[my_df['DayName'] == 'Wed', 'DayName'] = 3
my_df.loc[my_df['DayName'] == 'Thu', 'DayName'] = 4
my_df.loc[my_df['DayName'] == 'Fri', 'DayName'] = 5
my_df.loc[my_df['DayName'] == 'Sat', 'DayName'] = 6
my_df.loc[my_df['DayName'] == 'Sun', 'DayName'] = 7

my_df.loc[my_df['WorkType'] == 'WorkDay', 'WorkType'] = 1
my_df.loc[my_df['WorkType'] == 'DayOff', 'WorkType'] = 2
my_df.loc[my_df['WorkType'] == 'Holiday', 'WorkType'] = 3

my_df.loc[my_df['PrevWorkType'] == 'WorkDay', 'PrevWorkType'] = 1
my_df.loc[my_df['PrevWorkType'] == 'DayOff', 'PrevWorkType'] = 2
my_df.loc[my_df['PrevWorkType'] == 'Holiday', 'PrevWorkType'] = 3

# prepare learning set
categorial_cols = ['Year' ]
#one_hot_columns = ['Month', 'Day', 'DayName', 'WorkType', 'PrevWorkType', 'Time']
one_hot_columns = list()
prep_ls = Prepare_Ls(categorial_cols=categorial_cols, one_hot_encoding_names=one_hot_columns, response=response)
df_test = my_df.drop(['HistoryLoad', 'Id'], axis=1).copy()




numcols =  set(df_test.columns.values) - set(categorial_cols)
df_scale = df_test.loc[:, numcols]
cor = df_scale.corr()

# plt.matshow(cor)
# plt.xticks(range(len(df_scale.columns)), df_scale.columns)
# plt.yticks(range(len(df_scale.columns)), df_scale.columns)
# plt.colorbar()
# plt.show()

input_shape = prep_ls.get_nums_of_predictors(df=df_test) - 1
perc = 90
hid = [len(numcols)*3]  #round( (len(numcols)*perc)/100)
drop = [0.5]
# create neural model
nn = NNparams(hidden=hid, dropout=drop,
              optimizer=keras.optimizers.Adam(amsgrad=True),
              l1reg=0, l2reg=0,
              activation='relu', input_dim=input_shape,
              loss='mean_squared_error',
              train_metric=['mean_absolute_error'],
              batch_size=168,
              kernel_init='random_uniform', bias_init='zeros',
              compile=True
              )

# cross validation
cross_val = MyCv(model_cv_filepath=r'C:\Users\vgv\Desktop\PythonData\cv_weights.hdf5',
                 model_cv__final_filepath=r'C:\Users\vgv\Desktop\PythonData\cv_final_weights.hdf5',
                 path_to_initial_weigths=r'C:\Users\vgv\Desktop\PythonData\init_weigths.hdf5',
                 hidden=hid,
                 inp_shape=input_shape
                 )

first_pred = my_df.loc[my_df['Year'] == 2017 ]
first_id = first_pred.iloc[0].Id
last_pred_id = my_df.iloc[-1].Id


my_predict = My_Predict(my_df=my_df, nn=nn, response=response, cross_val=cross_val, prepare_ls=prep_ls)

my_predict.test_my_model(first_id = int(first_id), last_id = int(last_pred_id))





