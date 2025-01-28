import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection

class DataPrep(object):
  
    def __init__(self, raw_df: pd.DataFrame, categorical: list, log:list, mixed:dict, general:list, non_categorical:list, integer:list, type:dict, test_ratio:float):
        
        
        self.categorical_columns = categorical
        self.log_columns = log
        self.mixed_columns = mixed
        self.general_columns = general
        self.non_categorical_columns = non_categorical
        self.integer_columns = integer
        self.column_types = dict()
        self.column_types["categorical"] = []
        self.column_types["mixed"] = {}
        self.column_types["general"] = []
        self.column_types["non_categorical"] = []
        self.lower_bounds = {}
        self.label_encoder_list = []
        

        self.df = raw_df
        self.df = self.df.replace(r' ', np.nan)
       
        all_columns= set(self.df.columns)
        irrelevant_missing_columns = set(self.categorical_columns)
        relevant_missing_columns = list(all_columns - irrelevant_missing_columns)
        
        log_columns_numpy = self.df[self.log_columns].values # For faster computation
        lower_bounds = np.nanmin(log_columns_numpy, axis=0)
        eps = 1
        lower_transform_bound = np.where( #TODO:  COuld consider making this transform easier 
            lower_bounds > 0,
            0,  
            np.where(lower_bounds == 0, eps, -lower_bounds + eps)  
        )
        log_columns_numpy = np.log(log_columns_numpy + lower_transform_bound)
        self.lower_bounds = dict(zip(self.log_columns, lower_bounds))
        self.lower_transform_bound = lower_transform_bound # numpy form
        self.df[self.log_columns] = log_columns_numpy

        for i in relevant_missing_columns:
            if self.df[i].isnull().any():
                self.df[i] = self.df[i].fillna(-9999999)
                if i not in self.mixed_columns: # why do we drop them if they are log columns? Could be nice to have null values there
                    self.mixed_columns[i] = []
                self.mixed_columns[i].append(-9999999)
        
        
        for column_index, column in enumerate(self.df.columns):            
            if column in self.categorical_columns:        
                label_encoder = preprocessing.LabelEncoder()
                self.df[column] = self.df[column].astype(str)
                label_encoder.fit(self.df[column])
                current_label_encoder = dict()
                current_label_encoder['column'] = column
                current_label_encoder['label_encoder'] = label_encoder
                transformed_column = label_encoder.transform(self.df[column])
                self.df[column] = transformed_column
                self.label_encoder_list.append(current_label_encoder)
                self.column_types["categorical"].append(column_index)

                if column in self.general_columns:
                    self.column_types["general"].append(column_index)
            
                if column in self.non_categorical_columns:
                    self.column_types["non_categorical"].append(column_index)
            
            elif column in self.mixed_columns:
                self.column_types["mixed"][column_index] = self.mixed_columns[column]
            
            elif column in self.general_columns:
                self.column_types["general"].append(column_index)
            

        super().__init__()
        
    def inverse_prep(self, data, eps=1):
        
        data_pd = pd.DataFrame(data,columns=self.df.columns)

        for i in range(len(self.label_encoder_list)):
            column, label_encoder = self.label_encoder_list[i]["column"], self.label_encoder_list[i]["label_encoder"]
            le = self.label_encoder_list[i]["label_encoder"]
            data_pd[self.label_encoder_list[i]["column"]] = data_pd[self.label_encoder_list[i]["column"]].astype(int)
            data_pd[self.label_encoder_list[i]["column"]] = le.inverse_transform(data_pd[self.label_encoder_list[i]["column"]])

        
        log_columns_numpy = data_pd[self.log_columns].values
        log_columns_numpy = np.exp(log_columns_numpy) - self.lower_transform_bound
        data_pd[self.log_columns] = log_columns_numpy
        """
        if self.log_columns:
            for i in df:
                if i in self.log_columns:
                    lower_bound = self.lower_bounds[i]
                    if lower_bound>0:
                        df[i] = df[i].apply(lambda x: np.exp(x)) 
                    elif lower_bound==0:
                        df[i] = df[i].apply(lambda x: np.ceil(np.exp(x)-eps) if (np.exp(x)-eps) < 0 else (np.exp(x)-eps))
                    else: 
                        df[i] = df[i].apply(lambda x: np.exp(x)-eps+lower_bound)
        """
        
        
        if self.integer_columns:
            for column in self.integer_columns:
                data_pd[column]= (np.round(data_pd[column].values))
                data_pd[column] = data_pd[column].astype(int)

        data_pd.replace(-9999999, np.nan,inplace=True)
        data_pd.replace('empty', np.nan,inplace=True)

        return data_pd
