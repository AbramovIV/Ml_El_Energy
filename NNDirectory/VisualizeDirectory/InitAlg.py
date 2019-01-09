import keras
from NNDirectory.LsDirectory.LearningSetCl import LearningSet
from NNDirectory.LsDirectory.Prepare_Ls import Prepare_Ls
from NNDirectory.MyPredict import My_Predict
from NNDirectory.NNBuilderDirectory.MyCv import MyCv
from NNDirectory.NNBuilderDirectory.NNParams import NNparams
import matplotlib.pyplot as plt

# load data
ls_obj = LearningSet(path_to_df=r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt')
response = 'DiffHistoryLoad'
my_df = ls_obj.create_learningSet(ls_obj.initial_df)

# prepare learning set
categorial_cols = ['Year', 'Day', 'DayName', 'WorkType', 'PrevWorkType', 'Time']
one_hot_columns = ['Time', 'DayName', 'WorkType']
#one_hot_columns = list()
prep_ls = Prepare_Ls(categorial_cols=categorial_cols, one_hot_encoding_names=one_hot_columns, response=response)
df_test = my_df.drop(['HistoryLoad', 'Id'], axis=1)


numcols =  set(df_test.columns.values) - set(categorial_cols)
df_scale = df_test.loc[:, numcols]
cor = df_scale.corr()

#plt.matshow(cor)
#plt.xticks(range(len(df_scale.columns)), df_scale.columns)
#plt.yticks(range(len(df_scale.columns)), df_scale.columns)
#plt.colorbar()
#plt.show()

input_shape = prep_ls.get_nums_of_predictors(df=df_test) - 1
perc = 80
hid = round( (len(numcols)*perc)/100)
# create neural model
nn = NNparams(hidden=[hid], dropout=[0.0],
              optimizer=keras.optimizers.Adam(amsgrad=True),
              l1reg=0, l2reg=0,
              activation='relu', input_dim=input_shape,
              #loss='mean_squared_error',
              loss='logcosh',
              train_metric=['mean_absolute_error'],
              batch_size=168,
              kernel_init='random_uniform', bias_init='zeros'
              )

# cross validation
cross_val = MyCv(model_cv_filepath=r'C:\Users\vgv\Desktop\PythonData\cv_weights.hdf5',
                 model_cv__final_filepath=r'C:\Users\vgv\Desktop\PythonData\cv_final_weights.hdf5',
                 path_to_initial_weigths=r'C:\Users\vgv\Desktop\PythonData\init_weigths.hdf5'
                 )

first_pred_ind = my_df.index[my_df['Year'] == 2017][0]
first_pred_ind = first_pred_ind + 1
last_pred_ind = my_df.index[-1]

my_predict = My_Predict(my_df=my_df, nn=nn, response=response, cross_val=cross_val, prepare_ls=prep_ls)

my_predict.test_my_model(first_pred_ind=first_pred_ind, last_pred_ind=last_pred_ind)





