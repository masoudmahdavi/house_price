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
        #self.Describe = Describe(self.model, self.raw_house_dataframe)
        
    def __call__(self, hist:bool=False) -> Any:
        #self.Describe.describe_data(hist=hist)
        #self.Describe.data_visualization(base_map=True)
        pass

    def preprocess(self):
        handled_text_df = self.preprocess_data.text_encoder(self.raw_house_dataframe,
                                                            method='one_hot_encoder', # 'one_hot_encoder' or 'ordinal_encoder'
                                                            ) 
        
        print('Fix sparce df',handled_text_df)
        exit()
        cleaned_house_extended_df = self.preprocess_data.clean_miss_data(
                                                self.raw_house_dataframe, 
                                                clean_method='fill_miss'# 3 clean methods are available
                                                ) 
        
        house_extended_df = self.preprocess_data.combine_feature(self.raw_house_dataframe)
        normiaized_df = self.preprocess_data.norm_num_data(cleaned_house_extended_df, norm_method='min_max') #'min_max' or 'Standard'
        
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