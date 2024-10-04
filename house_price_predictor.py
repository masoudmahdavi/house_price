from typing import Any
from utils.data import house_dataframe
from model.model import Model
import matplotlib.pyplot as plt

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
if __name__ == "__main__":
    model = Model()
    house_price = HousePrice(model)
    house_price(hist=True)