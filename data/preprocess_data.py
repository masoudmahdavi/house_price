from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import scatter_matrix
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
                                    bins=[-2,-1,-0.5,0,0.5,1,2, np.inf], # np.inf means bigger than 6
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
        
        print('\ndistribution of district income in train data set :\n', train['income_cat'].value_counts().sort_index()/len(train))
        print('\ndistribution of district income in test data set :\n',test['income_cat'].value_counts().sort_index()/len(test))

        train = self._drop_income_cat(train, 'income_cat')
        test = self._drop_income_cat(test, 'income_cat')
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

    def _drop_income_cat(self, df:pd.DataFrame, column_name:str):
        df.drop(column_name, axis=1, inplace=True)
        return df
      
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
            

    def clean_miss_data(self, dataframe:pd.DataFrame, clean_method:str) -> pd.DataFrame:
        """Cleans data before any process from missing values

        Args:
            clean_method (str): Three options have to fix the missing values problem:
                                        1. Get rid of the corresponding districts(drop_rows).
                                        2. Get rid of the whole attribute(drop_column).
                                        3. Set the missing values to some value(fill_miss).

        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        
        if clean_method == "drop_rows":
                dataframe.dropna(inplace=True) # Option 1
        elif clean_method == "drop_column":
                columns_with_miss_vals = self.miss_val_columns(dataframe)
                dataframe = dataframe.drop(columns_with_miss_vals, axis=1) # Option 2
        elif clean_method == "fill_miss":
                dataframe = self.knn_imputer(dataframe) # Option 3 
                # dataframe = self.median_imuter(dataframe) # Median method

        return dataframe
      
    def miss_val_columns(self, dataframe:pd.DataFrame) -> list[str]:
        """Finds columns that contain missing values

        Args:
            dataframe (pd.DataFrame): Dataframe contains missing values

        Returns:
            list[str]: List of columns with miss values
        """
        columns_with_miss_vals = dataframe.columns[dataframe.isnull().any()].tolist()
        return(columns_with_miss_vals)

    def knn_imputer(self, dataframe:pd.DataFrame) -> pd.DataFrame:
        """Fills miss values of dataframe with knn method

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
      
    def median_imuter(self, dataframe:pd.DataFrame) -> pd.DataFrame:
        """Fills miss values of dataframe with median values of column

        Args:
            dataframe (pd.DataFrame): Dataframe contains missing values

        Returns:
            pd.DataFrame: Dataframe contains no missing values
        """
        meadian = dataframe["total_bedrooms"].median()
        dataframe["total_bedrooms"].fillna(meadian, inplace=True) 

        return dataframe

    def text_encoder(self, dataframe:pd.DataFrame, method='one_hot_encoder') -> pd.DataFrame:
        """Handels texts in dataframe.

        Args:
            dataframe (pd.DataFrame): The dataframe with text columns
            method (str, optional): Method of how to counter with texts. 
                                    Defaults to 'one_hot_encoder'.

        Returns:
            pd.DataFrame: Dataframe withouadd_encoded_to_dft text format columns.
        """
        ocean_prox_cat = dataframe[["ocean_proximity"]]
        if method == 'one_hot_encoder':
                encoded_df = self.one_hot_encoder(ocean_prox_cat)
                encoded_df = self.add_encoded_to_df(encoded_df, dataframe)
                
        elif method == 'ordinal_encoder':
                encoded_df = self.ordinal_encoder(ocean_prox_cat)
                encoded_df = self.add_encoded_to_df(encoded_df, dataframe)
                
        return encoded_df

    def one_hot_encoder(self, ocean_prox_cat):
        one_hot_encoder = OneHotEncoder()
        encoded_cat = one_hot_encoder.fit_transform(ocean_prox_cat)
        csr_encoded_cat = pd.DataFrame(encoded_cat, columns=ocean_prox_cat.columns,
                                        index=ocean_prox_cat.index)
        return csr_encoded_cat

    def add_encoded_to_df(self, encoded_df:pd.DataFrame, dataframe:pd.DataFrame) -> pd.DataFrame:
        dataframe['ocean_proximity'] = encoded_df
        # dataframe['dense_matrix'] = dataframe['ocean_proximity'].apply(lambda x: x.toarray())
        return dataframe

    def ordinal_encoder(self, ocean_prox_cat):
        ordinal_encoder = OrdinalEncoder()
        encoded_cat = ordinal_encoder.fit_transform(ocean_prox_cat)
        csr_encoded_cat = pd.DataFrame(encoded_cat, columns=ocean_prox_cat.columns,
                                        index=ocean_prox_cat.index)
        return csr_encoded_cat
      
    def norm_num_data(self, num_dataframe:pd.DataFrame, norm_method:str):    
        data_labels = num_dataframe["median_house_value"].copy()
        num_dataframe = self._drop_income_cat(num_dataframe, "median_house_value")
        if norm_method == "min_max":
            min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
            norm_df = min_max_scaler.fit_transform(num_dataframe)
    
        elif norm_method == "Standard":
            std_scaler = StandardScaler()
            norm_df = std_scaler.fit_transform(num_dataframe)
        
        df = pd.DataFrame(norm_df, columns=num_dataframe.columns)
        df["median_house_value"] = data_labels
        return df