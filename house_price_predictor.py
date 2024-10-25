from typing import Any
from utils.load_data import house_dataframe
from model.model import Model
from utils.describe_data import Describe
from data.preprocess_data import PreProcessData
from utils.load_data import house_dataframe

import pandas as pd



class HousePrice:
    def __init__(self, model:Model) -> None:
        self.model = model
        self.house_dataframe = house_dataframe(self.model)
        self.preprocess_data = PreProcessData(self.model)
        self.Describe = Describe(self.model, self.house_dataframe)
        
    def __call__(self, hist:bool=False) -> Any:
        # self.Describe.describe_data(hist=hist)
        self.Describe.data_visualization(base_map=True)

    def preprocess(self):
        train, test = self.preprocess_data.stratum_income(self.house_dataframe, split_n=10, hist=True)
        

if __name__ == "__main__":
    model = Model()
    house_price = HousePrice(model)
    house_price(hist=True)
    #house_price.preprocess()