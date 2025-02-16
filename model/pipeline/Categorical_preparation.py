import numpy as np
from sklearn import preprocessing

class Categorical_preparation:
    def __init__(self, column):
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit(column)
        
    def transform(self, column):
      
        encoded_column = self.label_encoder.transform(column)
        return np.asarray(encoded_column,int)

    def inverse_transform(self, column):
        column = np.asarray(column,int)
        return self.label_encoder.inverse_transform(column)

    def get_label_encoded(self, conditioning_value):
        # returns the encoded value of the real condition value
        return self.label_encoder.transform([conditioning_value])[0]