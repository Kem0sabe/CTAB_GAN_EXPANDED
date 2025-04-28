import numpy as np
import torch
from .column_transformer import Column_transformer

class General_transformer(Column_transformer):
    def __init__(self, column):
        super().__init__()
        self.min = column.min()
        self.max = column.max()

        self.fit(column.to_numpy())

    def fit(self, data_col):
        self.model = None 
        self.components = None
        self.output_info = [(1, 'tanh','yes_g')]
        self.output_dim = 1

    def transform(self, data_col):
        
        current = (data_col - (self.min)) / (self.max  - self.min) * 2 - 1
        current = current.reshape([-1, 1])
        return current

    def inverse_transform(self, data,st):
        u = data[:, st]
        u = (u + 1) / 2
        u = np.clip(u, 0, 1)
        u = u * (self.max - self.min) + self.min
        new_st = st + 1
        return u, new_st, []

    def inverse_transform_static(self, data, transformer, st, device, n_clusters=10): #TODO: remove n cluster when finding out how to handle it
        u = data[:, st]
        u = (u + 1) / 2
        u = torch.clamp(u, 0, 1)
        u = u * (transformer.max - transformer.min) + transformer.min
        new_st = st + 1
        return u, new_st, []
        

        
        

