import numpy as np
import pandas as pd
from loess import loess_1d
import matplotlib.pyplot as plt
from DataDirectory.LoadData import LoadData


class MyDataCleasing:
    my_data = None
    dfToClean = None
    initDf = None
    def __init__(self, data_path):
        self.my_data = LoadData(data_path)

    def prepare_initdf(self):
        # remove unnecessary columns in initial df
        self.my_data.initDf = self.my_data.initDf.drop('LoadPlan', axis=1)
        # change values of electricity load that <= 0 to nan, i.e. a priory outliers
        self.my_data.initDf.loc[self.my_data.initDf['HistoryLoad'] <= 0, 'HistoryLoad'] = np.nan
        self.initDf = self.my_data.initDf.copy()
        print('count of nan = ', self.my_data.initDf['HistoryLoad'].isna().sum())
        # create diff el with lag = 1 load column in df
        self.my_data.initDf['DiffElLoad'] = self.my_data.initDf.HistoryLoad.diff()
        # filling df that will be cleaned from other outliers
        self.dfToClean = self.my_data.initDf.copy().dropna()

        self.dfToClean = self.dfToClean.set_index('Id')
        mdc.dfToClean['Id'] = self.dfToClean.index.values

    def find_outlires(self):
        # clear by work types, working and other
        work_types_flags = ['WorkDay', 'no']
        countOfReplace = 0
        # alg for clear outliers
        for wt in work_types_flags:
            # select new df through current worktype
            if wt == work_types_flags[0]:
                workingDays = self.dfToClean.loc[self.dfToClean['WorkType'] == work_types_flags[0], :]
            else:
                workingDays = self.dfToClean.loc[self.dfToClean['WorkType'] != work_types_flags[0], :]
            # clear in all years and by current year
            years = workingDays.Year.unique()

            for yr in years:
                myDf = workingDays.copy().loc[workingDays['Year'] == yr, :]
                # clear by current month in year
                months = myDf['Month'].unique()
                for mn in months:
                    # select df with current month in current year
                    my_df_current_month_in_year = myDf.loc[myDf['Month'] == mn, :]
                    # get results from loess fitting
                    xout, yout, weigts = loess_1d.loess_1d(my_df_current_month_in_year['Time'].values,
                                                           my_df_current_month_in_year['DiffElLoad'].values, frac=0.2)
                    # create column with loess result
                    my_df_current_month_in_year['LoessSm'] = yout
                    # calc resudials of fitting from initial data
                    resudials = my_df_current_month_in_year['DiffElLoad'].values - \
                                my_df_current_month_in_year['LoessSm'].values
                    # create column with resudials in current df
                    my_df_current_month_in_year['Resudials'] = resudials
                    # create confideince interval
                    # get id values
                    id_vec = my_df_current_month_in_year.index.tolist()
                    # get lower and higher quantilies
                    qL, qH = np.percentile(my_df_current_month_in_year['Resudials'], [15, 85])
                    # iqr interval
                    my_iqr = qH - qL
                    # coeff to iqr interval
                    coef_conf = 2.0
                    # lower bound of conf interval
                    lower_conf = qL - (coef_conf * my_iqr)
                    # top bound of conf interval
                    top_conf = qH + (coef_conf * my_iqr)
                    # now replace outlier, i.e. value that outside conf interval
                    for k in id_vec:
                        # search outliers
                        #current candidate
                        candidate = my_df_current_month_in_year.ix[k, 'Resudials']
                        if candidate < lower_conf or candidate > top_conf:
                            self.dfToClean.ix[k, 'DiffElLoad'] =\
                                my_df_current_month_in_year.ix[k, 'LoessSm']
                            countOfReplace = countOfReplace + 1
        print('Count of all outliers = ', countOfReplace)

data_path = "C:/Users/vgv/Desktop/PythonData/initSet.txt"
mdc = MyDataCleasing(data_path=data_path)
mdc.my_data.fill_data()
mdc.prepare_initdf()
mdc.find_outlires()
# add skipped rows of initial df to cleaned

id_of_initDf = mdc.initDf['Id'].values
id_of_cleaned = mdc.dfToClean['Id'].values

ids_toadd = list(set(id_of_initDf) - set(id_of_cleaned))
print(ids_toadd)
for i in ids_toadd:
    mdc.dfToClean = mdc.dfToClean.append(mdc.initDf.loc[mdc.initDf['Id'] == i],
                                         ignore_index=True, verify_integrity=True)

mdc.dfToClean = mdc.dfToClean.sort_values(by=['Id'], axis=0)
#check max id difference
ids_diff = mdc.initDf['Id'].diff(1)
print('max id_diff = ', ids_diff.max())

mdc.dfToClean = mdc.dfToClean.set_index('Id')
mdc.dfToClean['Id'] = mdc.dfToClean.index.values

############################
mdc.initDf["DiffElLoad"] = mdc.initDf['HistoryLoad'].diff(1)
cleaned_df = mdc.initDf.copy()
counterReplace = 0
## id's indexis is in wrong order

cleaned_df = cleaned_df.set_index('Id')
cleaned_df['Id'] = cleaned_df.index.values

ids_list = cleaned_df['Id']
nancounters= 0
print('start replaceing')
for i in ids_list:
    if np.isnan(cleaned_df.ix[i, 'DiffElLoad']):
        nancounters = nancounters + 1
        print('count of nan = ', nancounters)
        continue
    if mdc.dfToClean.ix[i, 'DiffElLoad'] != cleaned_df.ix[i, 'DiffElLoad']:
        cleaned_df.ix[i, 'HistoryLoad'] = \
            cleaned_df.ix[i - 1, 'HistoryLoad'] + mdc.dfToClean.ix[i, 'DiffElLoad']
        counterReplace = counterReplace + 1



print('Was replaced: ', counterReplace)
plt.plot(mdc.my_data.initDf['HistoryLoad'].values)
plt.show()
plt.plot(cleaned_df['HistoryLoad'].values)
plt.show()
print('count of nan = ', cleaned_df['HistoryLoad'].isna().sum())
cleaned_df = cleaned_df.drop(['DiffElLoad'], axis=1)
cleaned_df.to_csv(r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt', header=True, index=None, sep=' ', mode='a',
                  na_rep='nan')

