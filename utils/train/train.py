from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from model.model import Model
import pandas as pd

class Train:
    def __init__(self, model:Model, df:pd.DataFrame):
        self.model = model
        self.train, self.train_labels = df[0]
        self.test, self.test_labels = df[1]


    def linear_regression(self):
        ml_model = TransformedTargetRegressor(LinearRegression(),
                        transformer=StandardScaler()
                   )
        
        ml_model.fit(self.train, self.train_labels)
        
        # print('samples: ',self.train_labels.iloc[:5].round(-2).values)
        # print('pred: ', predictions.round(-2))  
        # Does the same as the top code 
        # target_scaler = StandardScaler()
        # scaled_labels = target_scaler.fit_transform(self.train_labels.to_frame())
        # ml_model = LinearRegression()
        # ml_model.fit(self.train[["median_income"]], scaled_labels)
        # new_data = self.train[["median_income"]].iloc[:5]
        # true_val = self.train_labels.to_frame()[["median_house_value"]].iloc[:5]
        # scaled_prediction = ml_model.predict(new_data)
        # prediction = target_scaler.inverse_transform(scaled_prediction)

        return ml_model
    
    def decision_trees(self):
        ml_model = TransformedTargetRegressor(DecisionTreeRegressor(),
                        transformer=StandardScaler()
                   )
        ml_model.fit(self.train, self.train_labels)
        return ml_model
    
    def random_forest(self):
        ml_model = TransformedTargetRegressor(RandomForestRegressor(),
                        transformer=StandardScaler()
                   )
        ml_model.fit(self.train, self.train_labels)
        return ml_model