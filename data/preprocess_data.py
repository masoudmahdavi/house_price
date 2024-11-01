from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from model.model import Model

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

      def stratum_income(self, data:pd.DataFrame, n_strat_splits:int=1, hist:bool=False):
            data['income_cat'] = pd.cut(data['median_income'], 
                                        bins=[0,1,2,3,4,5,6, np.inf], # np.inf means bigger than 6
                                        labels=[0,1,2,3,4,5,6]) 
            
            if hist==True:
                  data['income_cat'].value_counts().sort_index().plot.bar(rot=0,grid=True)
                  plt.xlabel('Income category')
                  plt.ylabel('Number of districts')
                  plt.show()

            splitter = StratifiedShuffleSplit(n_splits=n_strat_splits, test_size=0.2, random_state=42)
            split_parts = []

            for train_split_index, test_split_index in splitter.split(data, data['income_cat']):
                  strat_train_set_n = data.iloc[train_split_index]
                  strat_test_set_n = data.iloc[test_split_index]
                  split_parts.append([strat_train_set_n, strat_test_set_n])
            
            train:pd.DataFrame = split_parts[0][0]
            test:pd.DataFrame = split_parts[0][1]
            print(train['income_cat'].value_counts().sort_index()/len(train))
            print(test['income_cat'].value_counts().sort_index()/len(test))
            train, test = self._drop_income_cat(train, test)
            train, train_label = self.split_label(train)
            test, test_label = self.split_label(test)
            return (train, train_label), (test, test_label)
      
      def split_label(self, data:pd.DataFrame) -> tuple[pd.DataFrame]:
            """Splits data to labels and samples

            Args:
                data (pd.DataFrame): Data entered

            Returns:
                tuple[pd.DataFrame]: samlples and labels in tuple 
            """
            data_without_label = data.drop(["median_house_value"], axis=1)
            data_labels = data["median_house_value"].copy()
            return data_without_label, data_labels

      def _drop_income_cat(self, train:pd.DataFrame, test:pd.DataFrame):
            for part in (train, test):
                  part.drop('income_cat', axis=1, inplace=True)
            return train, test
      
      def description_correlation(self, dataframe:pd.DataFrame, plot:bool=False):
            corr_matrix = dataframe.corr()
            corr_matrix["median_house_value"].sort_values(ascending=False)
            print('------------------------------------------------')
            print('correlation matrix after combining feature: \n')
            print(corr_matrix)
            print('------------------------------------------------')
            if plot:
                  attributes = ["median_house_value", 
                                "median_income",
                                "housing_median_age",
                                "rooms_per_house",
                                "bedrooms_ratio",
                                ]
                  scatter_matrix(dataframe[attributes], figsize=(19, 16))
                  plt.show()

            

      def combine_feature(self, df:pd.DataFrame) -> pd.DataFrame:
            df["rooms_per_house"] = df["total_rooms"] / df["households"]
            df["bedrooms_ratio"] = df["total_bedrooms"] / df["total_rooms"]
            df["people_per_house"] = df["population"] / df["households"]
            test_df = df.copy()
            houseing_num = test_df.select_dtypes(include=[np.number])
            self.description_correlation(houseing_num, plot=True)
            return df
            

      def clean_dataframe(self, dataframe:pd.DataFrame, clean_option:str) -> pd.DataFrame:
            """Cleans data before any process from missing values

            Args:
                clean_option (str): Three option has to fix missing values problem:
                                          1. Get rid of the corresponding districts.
                                          2. Get rid of the whole attribute.
                                          3. Set the missing values to some value.

            Returns:
                pd.DataFrame: Cleaned dataframe
            """
           
            if clean_option == "drop_rows":
                  dataframe.dropna(subset=["total_bedrooms"], inplace=True) # Option 1
            elif clean_option == "drop_column":
                  dataframe.drop("total_bedrooms", axis=1) # Option 2
            elif clean_option == "fill_miss":
                  dataframe = self.knn_imputer(dataframe)
                  # meadian = dataframe["total_bedrooms"].median()
                  # dataframe["total_bedrooms"].fillna(meadian, inplace=True) # Option 3 

            return dataframe
      
      def knn_imputer(self, dataframe:pd.DataFrame) -> pd.DataFrame:
            """Fills miss values of dataframe 

            Args:
                dataframe (pd.DataFrame): Dataframe contains missing values

            Returns:
                pd.DataFrame: Dataframe contains no missing values
            """
            imputer = KNNImputer(n_neighbors=5)
            dataframe = dataframe.select_dtypes(include=[np.number])
            filled_numpy = imputer.fit_transform(dataframe)
            filled_dataframe = pd.DataFrame(filled_numpy, columns=dataframe.columns,
                                            index=dataframe.index)

            return filled_dataframe
      
      def one_hot_encoder(self, dataframe:pd.DataFrame):
            one_hot_encoder = OneHotEncoder()
            ocean_proximity_cat = dataframe[["ocean_proximity"]]
            encoded_cat = one_hot_encoder.fit_transform(ocean_proximity_cat)
            return encoded_cat

      def ordinal_encoder(self, dataframe:pd.DataFrame):
            ordinal_encoder = OrdinalEncoder()
            ocean_proximity_cat = dataframe[["ocean_proximity"]]
            encoded_cat = ordinal_encoder.fit_transform(ocean_proximity_cat)
            return encoded_cat
