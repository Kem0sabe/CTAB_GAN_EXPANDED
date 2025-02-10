import numpy as np
import pandas as pd
import torch
from typing import List


from model.transformer.Column_transformer import Column_transformer
from model.transformer.GMM_transformer import GMM_transformer
from model.transformer.Categorical_transformer import Categorical_transformer
from model.transformer.Mixed_data_transformer import Mixed_data_transformer
from model.transformer.Gaussian_transformer import Gaussian_transformer

class DataTransformer():
    
    def __init__(self, train_data=pd.DataFrame, categorical_list=[], mixed_dict={}, general_list=[], n_clusters=10, eps=0.005):
        self.meta = None
        self.n_clusters = n_clusters
        self.eps = eps
        self.train_data = train_data
        self.categorical_columns= categorical_list
        self.mixed_columns= mixed_dict
        self.general_columns = general_list

        self.fit()
        
    
    def fit(self):
        self.transformers = self.setup_transformers()
        return self.fit_transformers(self.train_data)  
    

    def setup_transformers(self) -> List[Column_transformer]: #TODO: add options for clusers and eps and other stuff
        transformers = []
        for index in range(self.train_data.shape[1]):
            column = self.train_data.iloc[:,index]
            if index in self.categorical_columns:
                transformer = Categorical_transformer(column)
            elif index in self.mixed_columns.keys():
                transformer = Mixed_data_transformer(column,self.mixed_columns[index])
            elif index in self.general_columns:
                transformer = Gaussian_transformer(column)
            else:
                transformer = GMM_transformer(column)
            transformers.append(transformer)
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
        for i in range(len(self.transformers)):
            transformer = self.transformers[i]
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
        general_columns = []
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


