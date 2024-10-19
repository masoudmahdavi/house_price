from model.model import Model
import pandas as pd
import matplotlib.pyplot as plt

class Describe:
      def __init__(self, model:Model, data:pd.DataFrame):
            self.model = model
            self.data = data
            
      def describe_data(self, hist):
            head = self.data.head()
            info = self.data.info()
            describe = self.data.describe()
            
            print('\nhead: ', head, '\n')
            print('\ninfo: ', info)
            print('\ndescribe: ', describe)
            if hist:
                  self.data.hist(bins=100, figsize=(12, 8))
                  plt.show()
                  # for column in self.data.select_dtypes(include=np.number).columns:
                  #     data = self.data[column].to_numpy()
                  
                  #     hist, bins = np.histogram(data, bins=100)
                  #     fig = tpl.figure()
                  #     fig.hist(hist, bins, orientation='vertical')

                  #     print(f"Histogram for {column}:")
                  #     fig.show()
      
      def data_visualization(self, base_map:bool=False):
            copy_data = self.data.copy()
            if base_map:
                  self.visualization_on_basemap()
            else:
                  self.local_visual(copy_data)
                  
            plt.show()

      @staticmethod
      def local_visual(data):
            data.plot(kind='scatter', x="longitude", y="latitude", grid=True)

      def visualization_on_basemap():
            pass