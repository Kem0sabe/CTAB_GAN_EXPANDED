import numpy as np
import pandas as pd
import torch
from typing import List
from ..pipeline.data_preparation import DataPrep

from .column_transformers.column_transformer import Column_transformer
from .column_transformers.gmm_transformer import GMM_transformer
from .column_transformers.categorical_transformer import Categorical_transformer
from .column_transformers.mixed_data_transformer import Mixed_data_transformer
from .column_transformers.gaussian_transformer import Gaussian_transformer
from .column_transformers.general_transformer import General_transformer
from .column_transformers.gaussian_truncated_transformer import Gaussian_truncated_transformer


class DataTransformer():
    
    def __init__(self, prepared_data, data_prep: DataPrep, categorical_list, mixed_dict, general_list,truncated_gaussian_columns, n_clusters=10, eps=0.005):
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps
        
        self.data_prep = data_prep
       
        self.transformers = self.setup_transformers(prepared_data,categorical_list, mixed_dict, general_list,truncated_gaussian_columns)
        

    

    
    def get_column_transform(self, data,column, categorical_list, mixed_dict, general_list,truncated_gaussian_columns):
        if column in categorical_list: return Categorical_transformer(data[column])
        if column in mixed_dict: return Mixed_data_transformer(data[column],mixed_dict[column])
        if column in general_list: return General_transformer(data[column])
        if column in truncated_gaussian_columns: return Gaussian_truncated_transformer(data[column])
        return GMM_transformer(data[column])  # No type specified by user


    def setup_transformers(self,data,categorical_list, mixed_dict, general_list,truncated_gaussian_columns = []):
        transformers = []
        for column in data.columns:
            column_type = self.get_column_transform(data,column, categorical_list, mixed_dict, general_list,truncated_gaussian_columns)
            transformers.append(column_type)
        return transformers


            
    def fit_transformers(self,data):
        assert self.transformers is not None, "Transformers are not initialized"
        assert len(self.transformers) == data.shape[1], "Mismatch between number of transformers and data columns"
        for i in range(len(self.transformers)):
            transformer = self.transformers[i]
            transformer.fit(data.iloc[:,i].to_numpy())
        return True


       
    def transform(self, data):
        transforms = []
        for i, transformer in enumerate(self.transformers):
            if transformer is None: continue
            transforms.append(transformer.transform(data.iloc[:,i].to_numpy()))

        return np.concatenate(transforms, axis=1)

    def inverse_transform(self, data):
        data_t = np.zeros([len(data), len(self.transformers)])
        all_invalid_ids = []
        st = 0
        for idx, transformer in enumerate(self.transformers):
            new_data, st, invalid_ids = transformer.inverse_transform(data, st)
            data_t[:,idx] = new_data
            all_invalid_ids += invalid_ids
        all_invalid_ids = np.unique(all_invalid_ids)
        data_t = np.delete(data_t, list(all_invalid_ids), axis=0)
        return data_t, len(all_invalid_ids)


    def inverse_transform_static(self,data, transformer, device,n_clusters=10):
        gaussian_columns = []
        transformers = transformer.get_transformers()
        data_t = torch.zeros(len(data), len(transformers), device=device)
        all_invalid_ids = []
        st = 0
        for id_, transformer in enumerate(transformers):
            new_data, st, invalid_ids = transformer.inverse_transform_static(data, transformer, st,device,n_clusters)
            data_t[:,id_] = new_data
            all_invalid_ids += invalid_ids

        mask = torch.ones(len(data_t), dtype=torch.bool, device=device)
        mask[list(all_invalid_ids)] = False


        data_t = data_t[mask]

        return data_t, len(all_invalid_ids)

    def get_transformers(self): # TODO: should be removed if possible with the inverse backporpagation
        return self.transformers
         
    def get_output_info(self):
        return [transformer.get_output_info() for transformer in self.transformers]

    def get_output_dim(self):
        return sum([transformer.get_output_dim() for transformer in self.transformers])

    def get_components(self): #TODO: check if needed
        return [transformer.get_components() for transformer in self.transformers]

   
            

    

class ImageTransformer():

    def __init__(self, side):
    
        self.height = side
            
    def transform(self, data):

        if self.height * self.height > len(data[0]):
            
            padding = torch.zeros((len(data), self.height * self.height - len(data[0]))).to(data.device)
            data = torch.cat([data, padding], axis=1)

        return data.view(-1, 1, self.height, self.height)

    def inverse_transform(self, data):
        
        data = data.view(-1, self.height * self.height)

        return data


