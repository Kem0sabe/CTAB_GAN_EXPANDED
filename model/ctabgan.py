"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from .pipeline.data_type_assigner import Data_type_assigner
from .pipeline.data_preparation import DataPrep
from .pipeline.Column_assigner import Column_assigner, Transform_type
from .synthesizer.ctabgan_synthesizer import CTABGANSynthesizer


import warnings
import numpy as np

warnings.filterwarnings("ignore")

class CTAB_XTRA_DP():

    def __init__(self,
                 df,
                 test_ratio = 0.20,
                 categorical_columns = [], 
                 log_columns = [],
                 mixed_columns= {},
                 gaussian_columns = [],
                 non_categorical_columns = [],
                 integer_columns = [],
                 problem_type = ("Classification",'workclass'),
                 dp_constraints = {
                    "epsilon_budget": 10,
                    "delta": None,
                    "sigma": None,
                    "clip_coeff": 1
                 }
                 ):

        self.__name__ = 'CTAB_XTRA_DP'
              
        self.synthesizer = CTABGANSynthesizer()
        self.raw_df = df
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.gaussian_columns = gaussian_columns
        self.integer_columns = integer_columns

        self.problem_type = problem_type
        self.dp_constraints = dp_constraints

                
    def fit(self,epochs = 100):
        
        start_time = time.time()
        
     
        self.data_type_assigner = Data_type_assigner(self.raw_df, self.integer_columns)


       

        self.raw_df = self.data_type_assigner.assign(self.raw_df)

        self.data_prep = DataPrep(self.raw_df, self.categorical_columns, self.log_columns)

        self.prepared_data = self.data_prep.preprocesses_transform(self.raw_df)
        


        self.synthesizer.fit(self.prepared_data , self.data_prep, self.dp_constraints, self.categorical_columns, self.mixed_columns, self.gaussian_columns, self.problem_type,epochs)
        return
        


    def generate_samples(self,n=100,conditioning_column = None,conditioning_value = None):
        column_index = None
        column_value_index = None
        if conditioning_column and conditioning_value:
            column_index = self.prepared_data.columns.get_loc(conditioning_column) if conditioning_column in self.prepared_data.columns else ValueError("Conditioning column", conditioning_column, "not found in the data columns")
            column_value_index = self.data_prep.get_label_encoded(column_index, conditioning_value)

        sample_transformed = self.synthesizer.sample(n, column_index, column_value_index)
        sample_transformed = pd.DataFrame(sample_transformed, columns=self.prepared_data.columns)
        
        sample = self.data_prep.preprocesses_inverse_transform(sample_transformed)
        sample_with_data_types = self.data_type_assigner.assign(sample)
        return sample_with_data_types
        
        
  

    def generate_samples_index(self,n=100,index=None):

        sample = self.synthesizer.sample(n,0,index)
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df
