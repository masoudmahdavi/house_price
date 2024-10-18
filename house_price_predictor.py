from typing import Any
from utils.load_data import house_dataframe
from model.model import Model
from data.preprocess_data import PreProcessData
import matplotlib.pyplot as plt
import pandas as pd


class HousePrice:
    def __init__(self, model:Model) -> None:
        self.model = model
        self.house_dataframe = house_dataframe(self.model)
        self.preprocess_data = PreProcessData(self.model)
        
    def __call__(self, hist:bool=False) -> Any:
        head = self.house_dataframe.head()
        # info = self.house_dataframe.info()
        describe = self.house_dataframe.describe()
        
        # print('\nhead: ', head, '\n')
        # print('\ninfo: ', info)
        # print('\ndescribe: ', describe)
        if hist:
            self.house_dataframe.hist(bins=100, figsize=(12, 8))
            plt.show()
            # for column in self.house_dataframe.select_dtypes(include=np.number).columns:
            #     data = self.house_dataframe[column].to_numpy()
                
            #     hist, bins = np.histogram(data, bins=100)
            #     fig = tpl.figure()
            #     fig.hist(hist, bins, orientation='vertical')

            #     print(f"Histogram for {column}:")
            #     fig.show()
    def preprocess(self):
        train, test = self.preprocess_data.stratum_income(self.house_dataframe, split_n=10, hist=False)
        
        
        

if __name__ == "__main__":
    model = Model()
    house_price = HousePrice(model)
    house_price(hist=False)
    house_price.preprocess()