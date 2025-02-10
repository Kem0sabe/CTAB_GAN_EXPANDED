import numpy as np

from model.transformer.Column_transformer import Column_transformer

class Categorical_transformer(Column_transformer):
    def __init__(self, column):
        super().__init__()
        mapper = column.value_counts().index.tolist()
        self.size = len(mapper)
        self.i2s = mapper

    def fit(self, data_col):
        self.model = None
        self.components = None
        self.output_info = [(self.size, 'softmax', None)]
        self.output_dim = self.size
        

    def transform(self, data_col):
        self.ordering = None
        #col_t = np.zeros([len(data), info['size']])
        #idx = list(map(info['i2s'].index, current))
        #col_t[np.arange(len(data)), idx] = 1
        col_t = np.zeros([len(data_col), self.size])
        col_t[np.arange(len(data_col)), data_col] = 1
        return col_t
    def inverse_transform(self, data,st):
        u = data[:, st:st + self.size]
        idx = np.argmax(u, axis=1)
        #data_t[:, id_] = list(map(info['i2s'].__getitem__, idx))
        #data_t[:, id_] = list(map(info['i2s'].__getitem__, current))
        new_st = st + self.size
        return idx, new_st, []