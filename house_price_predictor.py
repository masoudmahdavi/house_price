from typing import Any
from utils.load_data import house_dataframe
from model.model import Model
from utils.describe_data import Describe
from data.preprocess_data import PreProcessData
from utils.load_data import house_dataframe
from utils.combine_df import combine_norm_and_text
from utils.train.train import Train
from utils.predict.predict import Predict
from utils.eval.evaluation import Evaluation

import pandas as pd



class HousePrice:
    def __init__(self, model:Model) -> None:
        self.model = model
        self.raw_house_dataframe = house_dataframe(self.model)
        self.preprocess_data = PreProcessData(self.model)
        #self.Describe = Describe(self.model, self.raw_house_dataframe)
        
        self.predict = Predict(self.model)
        self.eval = Evaluation()
        
    def __call__(self, hist:bool=False) -> Any:
        #self.Describe.describe_data(hist=hist)
        #self.Describe.data_visualization(base_map=True)
        pass

    def preprocess(self) -> tuple[pd.DataFrame]:
        handled_text_df = self.preprocess_data.text_encoder(self.raw_house_dataframe,
                                method='one_hot_encoder', # 'one_hot_encoder' or 'ordinal_encoder'
                          ) 
        
        cleaned_house_extended_df = self.preprocess_data.clean_miss_data(
                                        handled_text_df, 
                                        clean_method='fill_miss'# 3 clean methods are available
                                    ) 
        house_extended_df = self.preprocess_data.combine_feature(cleaned_house_extended_df)
        normalized_df = self.preprocess_data.norm_num_data(house_extended_df, norm_method='Standard') #'min_max' or 'Standard'
        combined_normiaized_text_df = combine_norm_and_text(normalized_df, handled_text_df)
        df = self.preprocess_data.stratum_income(combined_normiaized_text_df, n_strat_splits=10, hist=False)
        df = self.preprocess_data.convert_csr_to_dense_matrix(df)
        return df
        
    def train(self, df:tuple):
        self.train_model = Train(self.model, df)
        # lin_reg_model = self.train_model.linear_regression()
        # decision_trees_model = self.train_model.decision_trees()
        random_forest_model = self.train_model.random_forest()
        return random_forest_model
        
    def test(self, ml_model, test, test_lable):
        test_predictions = ml_model.predict(test)
        rmse = self.eval.eval(test_lable, test_predictions)
        confidence = 0.95
        confident_range = self.eval.confident(test_lable, test_predictions, confidence)
        print(f'Confident range({confidence}): ', confident_range)
        print('test RMSE: ', rmse)

    def prediction(self, ml_model, data:pd.DataFrame, data_lables):    
        prediction = self.predict.predict(ml_model, data)
        rmse = self.eval.cross_validation(data_lables, prediction, ml_model)
        return prediction

       
if __name__ == "__main__":
    model = Model()
    house_price = HousePrice(model)
    house_price(hist=True)
    df = house_price.preprocess()
    ml_model = house_price.train(df)
    train = df[0][0]
    test = df[1][0]
    train_label = df[0][1]
    test_lable = df[1][1]
    sample_data  = train
    sample_label_data = train_label.values

    house_price.test(ml_model, test, test_lable)
    prediction = house_price.prediction(ml_model, sample_data, sample_label_data)
    print(prediction)
    print(sample_label_data)
    print('---------------------------------------')
    print('\n\nPredict process has been finished.\n')