import pandas as pd

#function custom drop duplicates
def remove_duplicates(data:pd.DataFrame):
   return data.drop_duplicates() 

