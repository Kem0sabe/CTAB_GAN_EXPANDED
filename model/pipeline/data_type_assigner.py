
import warnings

class Data_type_assigner:
    def __init__(self, data,integer_columns = []):
    
        self.data_type = data.dtypes
        self.integer_columns = integer_columns + data.select_dtypes(include=['int64']).columns.tolist()
        for column in integer_columns:
            if column not in self.data_type: 
                warnings.warn(f"Column {column} not found in data, ignoring integer assignment")
                continue
            if data[column].isna().any(): continue # If we have nan values, we cant assign the column to interger, however we still treat it as integer
            data[column] = data[column].astype(int)
        
        self.number_of_decimals = self.get_column_desimal(data)
        

        

    def assign(self, data):
        for i in range(len(data.columns)):
            if self.number_of_decimals[i] is not None:
                data.iloc[:, i] = data.iloc[:, i].round(self.number_of_decimals[i])
      
        data = data.astype(self.data_type)
        return data

    
    def get_column_desimal(self, data):
        decimals = [None] * len(data.columns)
        for i in range(len(data.columns)):
            if data.iloc[:, i].dtype == 'float64':
                decimals[i] = data.iloc[:, i].apply(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0).max()

        for column in self.integer_columns: # The integers columns have 0 decimals by default
            idx = data.columns.get_loc(column)
            decimals[idx] = 0
        return decimals
