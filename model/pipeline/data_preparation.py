import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection

from .Log_preparation import Log_preparation
from .Categorical_preparation import Categorical_preparation

class DataPrep(object):
  
    def __init__(self, raw_df: pd.DataFrame, categorical_list, log_list):
        
    
        self.preprossesers =  self.setup_preprocesses(raw_df,categorical_list, log_list)
        super().__init__()

    
    def setup_preprocesses(self, data, categorical_list, log_list):
        
        preprocesses = []
        for column in data.columns:
            column_type = self.get_preprocesses_transform(data,column, categorical_list, log_list)
            preprocesses.append(column_type)
        return preprocesses


    def get_preprocesses_transform(self, data,column, categorical_list, log_list):
        if column in categorical_list: return Categorical_preparation(data[column])
        if column in log_list: return Log_preparation(data[column])
        return None  # No transformation specified

    
    def preprocesses_transform(self,data):
        assert len(self.preprossesers) == data.shape[1]
        data = data.copy()
        for idx, preprosseser in enumerate(self.preprossesers):
            if preprosseser is None: continue
            column = data.iloc[:,idx]
            data.iloc[:,idx] = preprosseser.transform(column)
            
        return data

    def preprocesses_inverse_transform(self,data):
        assert len(self.preprossesers) == data.shape[1]
        data = data.copy()
        for idx, preprosseser in enumerate(self.preprossesers):
            if preprosseser is None: continue
            column = data.iloc[:,idx]
            data.iloc[:,idx] = preprosseser.inverse_transform(column)
            
        return data

    def get_label_encoded(self,column_index, conditioning_value):
        # Takes in column index and find the encoded values of the conditioning value
        preprosess = self.preprossesers[column_index]
        if not isinstance(preprosess, Categorical_preparation): raise ValueError("Column is not categorical")
        return preprosess.get_label_encoded(conditioning_value)
        



    
                
  
        
