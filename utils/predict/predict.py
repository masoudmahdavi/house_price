
class Predict:
    def __init__(self, model):
        self.model = model
    
    def predict(self, ml_model, data):
        predictions = ml_model.predict(data)
        return predictions

