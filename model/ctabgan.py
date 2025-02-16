"""
Generative model training algorithm based on the CTABGANSynthesiser

"""
import pandas as pd
import time
from model.pipeline.data_preparation import DataPrep
from model.pipeline.Column_assigner import Column_assigner, Transform_type
from model.synthesizer.ctabgan_synthesizer import CTABGANSynthesizer

from model.transformer.Categorical_transformer import Categorical_transformer

import warnings
import numpy as np

warnings.filterwarnings("ignore")

class CTABGAN():

    def __init__(self,
                 df,
                 test_ratio = 0.20,
                 categorical_columns = [ 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'], 
                 log_columns = [],
                 mixed_columns= {'capital-loss':[0.0],'capital-gain':[0.0]},
                 gaussian_columns = ["age"],
                 non_categorical_columns = [],
                 integer_columns = ['age', 'fnlwgt','capital-gain', 'capital-loss','hours-per-week'],
                 problem_type= {"Classification": "income"}):

        self.__name__ = 'CTABGAN'
              
        self.synthesizer = CTABGANSynthesizer()
        self.raw_df = df
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.gaussian_columns = gaussian_columns

        # we remove non_categorical option
        # self.non_categorical_columns = non_categorical_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
                
    def fit(self,epochs = 150):
        
        start_time = time.time()
        
        #preprocess_assignments = Column_assigner.assign_columns_preprocess(self.raw_df, self.categorical_columns, self.log_columns)
        #transform_assignments = Column_assigner.assign_column_transforms(self.raw_df, self.categorical_columns, self.mixed_columns, self.gaussian_columns)
        
        self.data_prep = DataPrep(self.raw_df, self.categorical_columns, self.log_columns)

        self.prepared_data = self.data_prep.preprocesses_transform(self.raw_df)
        #self.prepared_data = self.prepared_data.fillna(-9999999)
        


        self.synthesizer.fit(self.prepared_data , self.data_prep, self.categorical_columns, self.mixed_columns, self.gaussian_columns, epochs)
        return
        end_time = time.time()
        print('Finished training in',end_time-start_time," seconds.")


    def generate_samples(self,n=100,conditioning_column = None,conditioning_value = None):
        column_index = None
        column_value_index = None
        if conditioning_column and conditioning_value:
            column_index = self.prepared_data.columns.get_loc(conditioning_column) if conditioning_column in self.prepared_data.columns else ValueError("Conditioning column", conditioning_column, "not found in the data columns")
            column_value_index = self.data_prep.get_label_encoded(column_index, conditioning_value)

        sample = self.synthesizer.sample(n, column_index, column_value_index)
        sample = pd.DataFrame(sample, columns=self.prepared_data.columns)
        #sample.replace(-9999999, np.nan, inplace=True)
        return self.data_prep.preprocesses_inverse_transform(sample)
        
  

    def generate_samples_index(self,n=100,index=None):

        sample = self.synthesizer.sample(n,0,index)
        sample_df = self.data_prep.inverse_prep(sample)

        return sample_df
