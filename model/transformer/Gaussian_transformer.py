from model.transformer.Column_transformer import Column_transformer

class Gaussian_transformer(Column_transformer):
    def __init__(self, column):
        super().__init__()
        self.min = column.min()
        self.max = column.max()

    def fit(self, data_col):
        self.model = None # Consider if we should explicit decleare model none and components (mainly for readability)
        self.components = None
        self.output_info = [(1, 'tanh','yes_g')]
        self.output_dim = 1

    def transform(self, data_col):
        self.ordering.append(None)
                  
        #if id_ in self.non_categorical_columns: #Wtf is this?
        #    info['min'] = -1e-3
        #    info['max'] = info['max'] + 1e-3
        
        current = (data_col - (self.min)) / (self.max  - self.min) * 2 - 1
        current = current.reshape([-1, 1])
        return current

    def inverse_transform(self, data,st):
        u = data[:, st]
        u = (u + 1) / 2
        u = np.clip(u, 0, 1)
        u = u * (self.max - self.min) + self.min
        #if id_ in self.non_categorical_columns:
        #data_t[:, id_] = np.round(u)
        new_st = st + 1
        return u, new_st, []
        

        
        

