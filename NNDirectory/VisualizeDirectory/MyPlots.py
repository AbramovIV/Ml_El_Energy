

import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean



predicions_file_path = r'C:\Users\vgv\Desktop\PythonData\Predictions\predictions.txt'
pred_df = pd.read_table(predicions_file_path, sep=" ", header=0)

plt.plot(pred_df.Mape)
print('mean: ', mean(pred_df.Mape))

plt.axes().set_xlabel('prediction_point')
plt.axes().set_ylabel('MAPE %')#

#plt.hist(pred_df.Mape.values)♥
plt.show()

plt.plot(pred_df.HistoryLoad, 'b.-', pred_df.Prediction, 'r.-')

plt.show()

# df_good = pred_df.iloc[500:900, :]
# df_bad = pred_df.iloc[900: 1300, :]
# plt.figure(1)
# plt.plot(df_good.HistoryLoad.diff(), 'b.-', df_good.Prediction.diff(), 'r.-')
# plt.figure(2)
# plt.plot(df_bad.HistoryLoad.diff(), 'b.-', df_bad.Prediction.diff(), 'r.-')
# plt.show()

#print(abs(pred_df.HistoryLoad.values - pred_df.Prediction.values))