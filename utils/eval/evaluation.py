from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
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