from utils.data import house_dataframe
from model.model import Model

class HousePrice:
    def __init__(self, model:Model) -> None:
        self.model = model
        self.house_data_dataframe = house_dataframe(self.model)
    


if __name__ == "__main__":
    model = Model()
    house_price = HousePrice(model)