from data import house_csv
from mkdir import mkdir

class HousePrice:
    def __init__(self) -> None:
        self.house_data_csv = house_csv()
        mkdir()
    


if __name__ == "__main__":
    house_price = HousePrice()