import numpy as np
import pandas as pd
import glob
from model.ctabgan import CTABGAN
df = pd.read_csv("Real_Datasets/adult.csv")
#df = df.drop(columns=['Year','Model'])
df = df[['age', 'workclass', 'capital-gain']]
df = df.head(1000)
df.loc[:9, 'capital-gain'] = np.nan

synthesizer =  CTABGAN(df,
                 test_ratio = 0.20,
                 categorical_columns = ["workclass"], 
                 log_columns = {}, # Fuck this log, task for tommorrow
                 mixed_columns= {"capital-gain": [0,np.nan]},
                 gaussian_columns = ["age"],
                 non_categorical_columns = [],
                 integer_columns = ["age","capital-gain"]) 


synthesizer.fit(1)