import urllib.request
import os
import tarfile
from utils.make_dirs import make_data_dir
from model.model import Model
import pandas as pd


def house_dataframe(model:Model) -> pd.DataFrame:
    """Downloads the data and prepares it in a suitable format

    Args:
        model (Model): Some important values set in this object as an attribute.

    Returns:
        pandas.Dataframe: house prices 
    """
    model.tar_dir = os.path.join('data/house.tgz')
    tar_base_dir = os.path.dirname(model.tar_dir)
    csv_dir = os.path.join(tar_base_dir, 'house_csv/house.csv')
    if not os.path.exists(csv_dir):
        make_data_dir(tar_base_dir)
        download_data(model, tar_base_dir)
    if not os.path.exists(csv_dir):
        extract_tar()
    
    house_price_dataframe = data_csv2df(csv_dir)
     
    return house_price_dataframe

def download_data() -> None:
    """Downloads data
    """
    url = "https://github.com/ageron/data/raw/main/housing.tgz"
    urllib.request.urlretrieve(url, "house.tgz")

def extract_tar(model:Model, tar_base_dir:str) -> None:
    with tarfile.open(model.tar_dir) as tar:
            tar.extractall(path=tar_base_dir)

def data_csv2df(csv_dir:str) -> pd.DataFrame:
    """Read csv file as pandas DataFrame

    Args:
        csv_dir (str): csv directory

    Returns:
        pd.DataFrame: house price data in dataframe format
    """
    df = pd.read_csv(csv_dir)
    return df