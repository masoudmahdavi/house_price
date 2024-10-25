from shapely.geometry import Point
from branca.colormap import linear
from model.model import Model

import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
import folium


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
      
      def visualization_on_basemap(self, data:pd.DataFrame):
            crs = {'init':'EPSG:4326'}
            geometry = [Point(xy) for xy in zip (data['longitude'], data['latitude'])]
            geo_df = gpd.GeoDataFrame(data,
                                      crs=crs,
                                      geometry=geometry)
            center_map = geo_df.geometry.y.mean(), geo_df.geometry.x.mean()
            map = folium.Map(location=center_map, zoom_start=8)
            
            limit = geo_df.median_house_value.max() - geo_df.median_house_value.min()
            step = limit // 10
            labels= []
            for i in range(0,10):
                  labels.append(geo_df.median_house_value.min() + (i*step))
            labels += [np.inf]
            
            geo_df['house_value_cat'] = pd.cut(data['median_house_value'], 
                                        bins=[14000, 100000, 200000,300000,400000, 500009, np.inf], # np.inf means bigger than 6
                                        labels=[0,1,2,3,4,5])
            colormap = linear.YlOrRd_09.scale(0, 6)
            house_val_dict = geo_df.set_index('house_value_cat')['median_house_value']
            folium.GeoJson(
                  geo_df,
                  name = 'house price',
                  marker=folium.Circle(radius=300, fill_color="orange", fill_opacity=0.9, color="black", weight=0.8),
                  style_function= lambda x:{'color':colormap(x['properties']['house_value_cat']),'fillColor':colormap(x['properties']['house_value_cat'])}
                  
            ).add_to(map)
            map.save('map.html')
            map
            # map.show_in_browser()
      
      def style_function(self, feature):
            house_cat = feature['properties']['house_value_cat']
            
            colormap = linear.YlOrRd_09.scale(0, 6)
            markup = f"""
                        <div style="width: 10px;
                                    height: 10px;
                                    border: 1px solid black;
                                    border-radius: 5px;
                                    background-color: {colormap(house_cat)};
                                    fill_color: red">
                        </div> 
                  
            """
            return {"html": markup}