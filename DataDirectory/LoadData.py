import pandas as pd


class LoadData:
    dataPath = None
    initDf = None

    def __init__(self, data_path: str):
        self.dataPath = data_path
        self.fill_data()

    def fill_data(self):
        self.initDf = pd.read_table(self.dataPath, sep=" ", header=0, na_values='nan', keep_default_na=True)
