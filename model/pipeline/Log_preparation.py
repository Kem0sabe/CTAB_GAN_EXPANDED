import numpy as np
import pandas as pd

class Log_preparation:
    def __init__(self, column,eps=1):
        column_numpy = column.to_numpy()
        lower_value = np.nanmin(column_numpy)
        self.lower_bound = 0 if lower_value > 0 else (eps if lower_value == 0 else -lower_value + eps) # Set lower bound to ensure all values are positive
        

    def transform(self, column):
        column_numpy = column.to_numpy()
        return np.log(column_numpy + self.lower_bound)

    def inverse_transform(self, column):
        column_numpy = column.to_numpy()
        return np.exp(column_numpy) - self.lower_bound

