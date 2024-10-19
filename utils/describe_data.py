from model.model import Model
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
import contextily as ctx



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
                  self.visualization_on_basemap(copy_data)
            else:
                  self.local_visual(copy_data)
            plt.show()

      @staticmethod
      def local_visual(data:pd.DataFrame):
            data.plot(kind='scatter', x="longitude", y="latitude", grid=True)

      @staticmethod
      def visualization_on_basemap(data:pd.DataFrame):
            crs = {'init':'EPSG:4326'}
            geometry = [Point(xy) for xy in zip (data['longitude'], data['latitude'])]
            geo_df = gpd.GeoDataFrame(data,
                                      crs=crs,
                                      geometry=geometry)
            
            ax = geo_df.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
            
            ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)