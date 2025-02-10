import numpy as np
import pandas as pd
import glob
from model.ctabgan import CTABGAN
df = pd.read_csv("Real_Datasets/car.csv")
df = df.drop(columns=['Year','Model'])

synthesizer =  CTABGAN(df,
                 test_ratio = 0.20,
                 categorical_columns = ["Brand","Model","Fuel_Type","Transmission"], 
                 log_columns = [],
                 mixed_columns= {"Engine_Size": [0]},
                 general_columns = [],
                 non_categorical_columns = [],
                 integer_columns = [],
                 problem_type= {"Classification": 'Price'}) 

synthesizer.fit(1)