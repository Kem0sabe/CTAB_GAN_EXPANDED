from enum import Enum
import numpy as np


## TODO: not used, probabilly could be used when columsn are not specified and we have to check them
class Transform_type(Enum):
    CATEGORICAL = 1
    MIXED = 2
    GAUSSIAN = 3
    CONTINUOUS = 4


class Preprocess_type(Enum):
    CATEGORICAL = 1
    LOG = 2

class Column_assigner:
   


    @classmethod
    def check_column_transform_type(cls, data_col):
        # TODO: One could add some checks for something here at a later point
        # For now we just assume it is continuous and fit by a GMM
        if cls.is_categorical(data_col):
            raise ValueError("Column: ", data_col, "is categorical. Include it in the categorical list")
        return Transform_type.CONTINUOUS

    @classmethod
    def is_categorical(cls, data_col):
        # Implement some fancy check here
        return False
        return False # Implement some fancy check here

                

        

    