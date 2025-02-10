
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df_filtered = pd.read_csv('A_greater_B.csv')

import numpy as np
import pandas as pd
import glob
from model.ctabgan import CTABGAN

synthesizer =  CTABGAN(df_filtered,
                 test_ratio = 0.20,
                 categorical_columns = [], 
                 log_columns = [],
                 mixed_columns= {},
                 general_columns = [],
                 non_categorical_columns = [],
                 integer_columns = [],
                 problem_type= None) 

synthesizer.fit(30)

synthesizer.generate_samples(30)