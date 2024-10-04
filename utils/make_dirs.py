import os

def make_data_dir(data_dir:str) -> None:
    """Make essential direcotories

    Args:
        data_dir (str): directory of data
    """
    os.makedirs(data_dir, exist_ok=True)

