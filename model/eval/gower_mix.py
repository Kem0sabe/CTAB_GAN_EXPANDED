
import numpy as np

from scipy.sparse import issparse






def gower_distance(data_x, data_y=None, weight=None, mix_features=None):
    X = data_x
    Y = data_y if data_y is not None else data_x

    

    if not isinstance(X, np.ndarray): X = np.asarray(X)
    if not isinstance(Y, np.ndarray): Y = np.asarray(Y)

    if X.shape[1] != Y.shape[1]:
        raise ValueError(f"X and Y must have the same number of columns")
    n_cols = X.shape[1] # the number of columns for both X and Y
    
    # Set default weights if not provided
    if weight is None:
        weight = np.ones(n_cols)
    else:
        weight = np.asarray(weight)

    XY = np.concatenate((X, Y))
    XY = replace_nulls_with_placeholder(XY)
 
    if issparse(X) or issparse(Y): raise TypeError("Sparse matrices are not supported!") 

    numeric_array, numeric_mask = build_numeric_array_and_mask(XY)
    column_max, column_min, column_means, column_stds = compute_column_statistics(numeric_array, numeric_mask)
    column_range = column_max - column_min

    XY = np.where(
        numeric_mask,
        (numeric_array - column_min) / column_range,
        XY  #
    )

    n_rows_X = X.shape[0]
    X_proc = XY[:n_rows_X]
    Y_proc = XY[n_rows_X:] 
    n_rows_X = X_proc.shape[0]
    n_rows_Y = Y_proc.shape[0]
    
    numeric_mask = numeric_mask.astype(bool)
    numeric_mask_X = numeric_mask[:n_rows_X]
    numeric_mask_Y = numeric_mask[n_rows_X:]

    total_weighted = np.zeros((n_rows_X, n_rows_Y))

    for i in range(n_rows_X):
        xi = X_proc[i]
        i_numeric_mask = numeric_mask_X[i]
        

        common_ones = np.logical_and(i_numeric_mask, numeric_mask_Y)
        common_zeros = np.logical_and(~i_numeric_mask, ~numeric_mask_Y)
        differing_values = np.logical_xor(i_numeric_mask, numeric_mask_Y)

        
        
        numerical_abs = np.abs(np.where(common_ones,Y_proc,np.nan) - np.where(common_ones,xi,np.nan))
        weighted_numeric_abs = numerical_abs * np.where(common_ones, weight, 0)
        weighted_numeric_abs_sum = np.nansum(weighted_numeric_abs, axis=1)
        
        categorical = np.where(
            common_zeros,
            xi != Y_proc,
            False
        )
        weighted_categorical = categorical * np.where(common_zeros, weight, 0)
        weighted_categorical_sum = np.nansum(weighted_categorical, axis=1)

        differing_values = differing_values * np.where(differing_values, weight, 0)
        weighted_differing_values_sum = np.nansum(differing_values, axis=1)

        total_weighted_sum = weighted_numeric_abs_sum + weighted_categorical_sum + weighted_differing_values_sum
        total_weighted_row = total_weighted_sum / np.sum(weight)

        total_weighted[i] = total_weighted_row

        
    return total_weighted
        




def build_numeric_array_and_mask(array):
    array = np.asarray(array, dtype=object)  # Handle mixed types safely
    n_rows, n_cols = array.shape

    numeric_array = np.zeros((n_rows, n_cols), dtype=float)
    mask = np.zeros((n_rows, n_cols), dtype=int)  # 0 = categorical by default


    for i in range(n_rows):
        for j in range(n_cols):
            val = array[i, j]

            if isinstance(val, (int, float)) and not isinstance(val, bool):
                numeric_array[i, j] = val
                mask[i, j] = 1
            elif isinstance(val, str):
                try:
                    float_value = float(val)
                    numeric_array[i, j] = float_value
                    mask[i, j] = 1
                except:
                    pass
            # else: keep it 0 (categorical)
    
    return numeric_array, mask



def compute_column_statistics(numeric_array, numeric_mask):
 
    column_max = []
    column_min = []
    column_means = []
    column_stds = []
    for col_idx in range(numeric_array.shape[1]):
        # Get the mask for this column
        col_mask = numeric_mask[:, col_idx]
        
        # Get the values for this column where mask is 1
        masked_values = numeric_array[:, col_idx][col_mask == 1]
        
        max_val = np.max(masked_values) if len(masked_values) > 0 else np.nan
        min_val = np.min(masked_values) if len(masked_values) > 0 else np.nan
        mean_val = np.mean(masked_values) if len(masked_values) > 0 else np.nan
        std_val = np.std(masked_values) if len(masked_values) > 0 else np.nan
        
        column_max.append(max_val)
        column_min.append(min_val)
        column_means.append(mean_val)
        column_stds.append(std_val)

    column_max = np.array(column_max)
    column_min = np.array(column_min)
    column_means = np.array(column_means)
    column_stds = np.array(column_stds)
    return column_max, column_min, column_means, column_stds




def is_null(val):
    return val is None or (isinstance(val, float) and np.isnan(val))

def replace_nulls_with_placeholder(matrix, placeholder="__MISSING__"):
    matrix = np.asarray(matrix, dtype=object)
    return np.array([
        [placeholder if is_null(x) else x for x in row]
        for row in matrix
    ], dtype=object)


