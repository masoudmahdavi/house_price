from model.model import Model
import pandas as pd
import matplotlib.pyplot as plt

class Describe:
      def __init__(self, model:Model):
            self.model = model
            
      def describe_data(self, data:pd.DataFrame, hist):
            head = data.head()
            info = data.info()
            describe = data.describe()
            
            print('\nhead: ', head, '\n')
            print('\ninfo: ', info)
            print('\ndescribe: ', describe)
            if hist:
                  data.hist(bins=100, figsize=(12, 8))
                  plt.show()
                  # for column in self.data.select_dtypes(include=np.number).columns:
                  #     data = self.data[column].to_numpy()
                  
                  #     hist, bins = np.histogram(data, bins=100)
                  #     fig = tpl.figure()
                  #     fig.hist(hist, bins, orientation='vertical')

                  #     print(f"Histogram for {column}:")
                  #     fig.show()