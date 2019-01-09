import pandas as pd
import matplotlib.pyplot as plt


initial_df = pd.read_table(r'C:\Users\vgv\Desktop\PythonData\initSet.txt', sep=" ", header=0, na_values='nan', keep_default_na=True)
history_init = initial_df['HistoryLoad']

cleaned_df = pd.read_table(r'C:\Users\vgv\Desktop\PythonData\cleanedDf.txt', sep=" ", header=0, na_values='nan', keep_default_na=True)
history_cleaned = cleaned_df['HistoryLoad']

counter_0 = 0
for h in history_init:
    if h <= 0:
        counter_0 = counter_0 + 1

dif = history_init - history_cleaned
un = pd.Series(dif).unique()
count = len(un)


plt.plot(dif)
plt.show()
z = 1