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
        self.raw_house_dataframe = house_dataframe(self.model)
        self.preprocess_data = PreProcessData(self.model)
        self.Describe = Describe(self.model, self.raw_house_dataframe)
        
    def __call__(self, hist:bool=False) -> Any:
        self.Describe.describe_data(hist=hist)
        self.Describe.data_visualization(base_map=True)

    def preprocess(self):
        one_hot_df = self.preprocess_data.one_hot_encoder(self.raw_house_dataframe) # 1. ordinal_encoder 2.one_hot_encoder
        house_extended_df = self.preprocess_data.combine_feature(self.raw_house_dataframe)
        cleaned_house_extended_df = self.preprocess_data.clean_dataframe(house_extended_df,
                                                                         clean_option='fill_miss')
        df = self.preprocess_data.stratum_income(cleaned_house_extended_df, n_strat_splits=10, hist=True)
        train, train_labels = df[0]
        test, test_labels = df[1]
        

if __name__ == "__main__":
    model = Model()
    house_price = HousePrice(model)
    house_price(hist=True)
    house_price.preprocess()
    print('---------------------------------------')
    print('\n\nPredict process has been finished.\n')