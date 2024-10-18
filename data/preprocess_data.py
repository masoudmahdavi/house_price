from model.model import Model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class PreProcessData:
      def __init__(self, model:Model) -> None:
            self.model = model

      def split_shuffle(self, data:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
            """Shuffles and splits data

            Args:
                data (pd.DataFrame): Data in dataframe format

            Returns:
                tuple[pd.DataFrame, pd.DataFrame]: Train and test parts 
            """
            train_set, test_set = train_test_split(data, test_size=0.2)
            return train_set, test_set

      def stratum_income(self, data:pd.DataFrame, split_n:int=1, hist:bool=False):
            data['income_cat'] = pd.cut(data['median_income'], 
                                        bins=[0,1,2,3,4,5,6, np.inf], # np.inf means bigger than 6
                                        labels=[0,1,2,3,4,5,6]) 
            
            if hist==True:
                  data['income_cat'].value_counts().sort_index().plot.bar(rot=0,grid=True)
                  plt.xlabel('Income category')
                  plt.ylabel('Number of districts')
                  plt.show()

            splitter = StratifiedShuffleSplit(n_splits=split_n, test_size=0.2, random_state=42)
            split_parts = []

            for train_split_index, test_split_index in splitter.split(data, data['income_cat']):
                  strat_train_set_n = data.iloc[train_split_index]
                  strat_test_set_n = data.iloc[test_split_index]
                  split_parts.append([strat_train_set_n, strat_test_set_n])
            
            train:pd.DataFrame = split_parts[0][0]
            test:pd.DataFrame = split_parts[0][1]
            print(train['income_cat'].value_counts().sort_index()/len(train))
            print(test['income_cat'].value_counts().sort_index()/len(test))
            train, test = self.drop_income_cat(train, test)

            return train, test
      
      def drop_income_cat(self, train:pd.DataFrame, test:pd.DataFrame):
            for part in (train, test):
                  part.drop('income_cat', axis=1, inplace=True)
            return train, test