

import matplotlib.pyplot as plt
import pandas as pd
from numpy import mean



predicions_file_path = r'C:\Users\vgv\Desktop\PythonData\Predictions\predictions.txt'
pred_df = pd.read_table(predicions_file_path, sep=" ", header=0)

plt.plot(pred_df.Mape[:300])
print('mean: ', mean(pred_df.Mape[:300]))
plt.axes().set_xlabel('prediction_point')
plt.axes().set_ylabel('MAPE %')
#plt.hist(pred_df.Mape.values)
plt.show()