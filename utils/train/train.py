from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
from model.model import Model

class Train:
    def __init__(self, model:Model):
        self.model = model

    def linear_regression(self, df:tuple):
    
        train, train_labels = df[0]
        test, test_labels = df[1]

        ml_model = TransformedTargetRegressor(LinearRegression(),
        transformer=StandardScaler())
        ml_model.fit(train[["median_income"]], train_labels)
        new_data = train[["median_income"]].iloc[:5]
        predictions = ml_model.predict(new_data)
        # print('samples: ',train_labels.iloc[:5].round(-2).values)
        # print('pred: ', predictions.round(-2))  
       
        # Does the same as the top code 
        # target_scaler = StandardScaler()
        # scaled_labels = target_scaler.fit_transform(train_labels.to_frame())
        # ml_model = LinearRegression()
        # ml_model.fit(train[["median_income"]], scaled_labels)
        # new_data = train[["median_income"]].iloc[:5]
        # true_val = train_labels.to_frame()[["median_house_value"]].iloc[:5]
        # scaled_prediction = ml_model.predict(new_data)
        # prediction = target_scaler.inverse_transform(scaled_prediction)

        return ml_model