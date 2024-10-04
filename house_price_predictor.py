from typing import Any
from utils.data import house_dataframe
from model.model import Model
import matplotlib.pyplot as plt
import termplotlib as tpl
import numpy as np

class HousePrice:
    def __init__(self, model:Model) -> None:
        self.model = model
        self.house_data_dataframe = house_dataframe(self.model)
    
    def __call__(self, hist:bool=False) -> Any:
        head = self.house_data_dataframe.head()
        info = self.house_data_dataframe.info()
        describe = self.house_data_dataframe.describe()
        

        print('\nhead: ', head, '\n')
        print('\ninfo: ', info)
        print('\ndescribe: ', describe)
        if hist:
            self.house_data_dataframe.hist(bins=100, figsize=(12, 8))
            plt.show()
            # for column in self.house_data_dataframe.select_dtypes(include=np.number).columns:
            #     data = self.house_data_dataframe[column].to_numpy()
                
            #     hist, bins = np.histogram(data, bins=100)
            #     fig = tpl.figure()
            #     fig.hist(hist, bins, orientation='vertical')

            #     print(f"Histogram for {column}:")
            #     fig.show()


if __name__ == "__main__":
    model = Model()
    house_price = HousePrice(model)
    house_price(hist=True)