import pandas as pd

def combine_norm_and_text(normed_df:pd.DataFrame, text_handled_df:pd.DataFrame) -> pd.DataFrame:
      text_handled_df = text_handled_df.loc[:, text_handled_df.columns=='ocean_proximity']
      normed_df['ocean_proximity'] = text_handled_df
      return normed_df
      