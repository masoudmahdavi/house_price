from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
from scipy import stats
import numpy as np
import pandas as pd

class Evaluation:
    def __init__(self):
        pass

    def eval(self, labels, predictions):
        rmse = root_mean_squared_error(labels,
                                           predictions,
                                           )
        print('RMSE lin_reg: ', rmse)
        return rmse
    
    def cross_validation(self, labels, predictions, ml_model):
        labels = (np.array(labels)).reshape(-1,1)
        predictions = (np.array(predictions)).reshape(-1,1)
        rmses = -cross_val_score(ml_model,
                                 labels,
                                 predictions,
                                 scoring="neg_root_mean_squared_error",
                                 cv=10,
                )
        
        print('\nPerformance: \n', pd.Series(rmses).describe())
        return rmses
    
    def confident(self, test_lable, test_predictions, confidence:float):
        confidence=confidence
        squared_errors = (test_predictions - test_lable) ** 2
        conf_range = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
            loc=squared_errors.mean(),
            scale=stats.sem(squared_errors)))
        print('confident')
        return conf_range