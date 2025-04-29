import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor ,HistGradientBoostingClassifier, HistGradientBoostingRegressor
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import warnings
from .gower_mix import gower_distance
from collections import defaultdict

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor


warnings.filterwarnings("ignore")





def get_summary_metrics(real,
                        fake,
                        categorical=[],
                        mixed={},
                        mnar=True,
                        problem="classification"):

    simmlarity = stat_sim(real,fake,categorical,mixed,mnar)
    if problem == "regression":
      models_to_run = ["dt","rf","hgb","cat","xgb"]
    elif problem == "classification":
      models_to_run = ["lr","dt","rf","xgb"]
    else:
      raise ValueError("Problem type must be either 'classification' or 'regression'")
    utility = get_utility_metrics(real,[fake],categorical=categorical, mixed=mixed,scaler="MinMax",problem=problem, models=models_to_run,test_ratio=.20)
    privacy = privacy_metrics(real,fake,metric='gower')
    return utility, simmlarity, privacy


def supervised_model_training(x_train, y_train, x_test, 
                              y_test, model_name,problem_type):
  
  model = get_supervised_model(model_name,problem_type)
  
  model.fit(x_train, y_train)
  pred = model.predict(x_test)

  if problem_type == "classification":
    acc = metrics.accuracy_score(y_test,pred)*100
    if len(np.unique(y_train))>2: # We have mulitclass classification
      predict = model.predict_proba(x_test)        
      auc = metrics.roc_auc_score(y_test, predict,average="weighted",multi_class="ovr")
      f1_score = metrics.precision_recall_fscore_support(y_test, pred,average="weighted")[2]

    else: # We have binary classification
      predict = model.predict_proba(x_test)[:,1]    
      auc = metrics.roc_auc_score(y_test, predict)
      f1_score = metrics.precision_recall_fscore_support(y_test,pred)[2].mean()
    return [acc, auc, f1_score] 
  
  if problem_type == "regression":
    mse = metrics.mean_absolute_percentage_error(y_test,pred)
    evs = metrics.explained_variance_score(y_test, pred)
    r2_score = metrics.r2_score(y_test,pred)
    return [mse, evs, r2_score]

  raise ValueError(f"Unknown problem type: {problem_type}. Supported types are 'classification' and 'regression'.")




def get_supervised_model(model_name,problem_type,random_state=42):
  if problem_type == "classification":
    if model_name == 'lr': return LogisticRegression(random_state=random_state,max_iter=500)
    if model_name == 'svm': return svm.SVC(random_state=random_state,probability=True)

    if model_name == 'dt': return tree.DecisionTreeClassifier(random_state=random_state)
    if model_name == 'rf': return RandomForestClassifier(random_state=random_state)
    if model_name == 'mlp': return MLPClassifier(random_state=random_state,max_iter=100)
    if model_name == 'xgb': return XGBClassifier(random_state=random_state, use_label_encoder=False, eval_metric='logloss')
    if model_name == 'lgbm': return LGBMClassifier(random_state=random_state,verbose=-1)
    if model_name == 'cat': return CatBoostClassifier(random_state=random_state, verbose=0)
    raise ValueError(f"Unknown model name: {model_name} for classification")
  
  if problem_type == "regression":
    if model_name == 'lin': return LinearRegression()
    if model_name == 'ridge': return Ridge(random_state=random_state)
    if model_name == 'lasso': return Lasso(random_state=random_state)
    if model_name == 'B_ridge': return BayesianRidge()
  
    if model_name == 'dt': return tree.DecisionTreeRegressor(random_state=random_state)
    if model_name == 'rf': return RandomForestRegressor(random_state=random_state)
    if model_name == 'hgb': return HistGradientBoostingRegressor(random_state=random_state)
    if model_name == 'xgb': return XGBRegressor(random_state=random_state)
    if model_name == 'lgbm': return LGBMRegressor(random_state=random_state,verbose=-1)
    if model_name == 'cat': return CatBoostRegressor(random_state=random_state, verbose=0)
    raise ValueError(f"Unknown model name: {model_name} for regression")

  raise ValueError(f"Unknown problem type: {problem_type}. Supported types are 'classification' and 'regression'.")




def split_data(df, test_ratio=.20, target=None,problem="classification"):
  if problem == "classification":
    if target is None: raise ValueError("Target column must be specified for classification")
    train, test = model_selection.train_test_split(df, test_size=test_ratio, stratify=df[target], random_state=42)
  else:
    train, test = model_selection.train_test_split(df, test_size=test_ratio, random_state=42)
  return train, test


def get_utility_metrics(real,
                        fakes,
                        categorical=[],
                        mixed={},
                        target=None,
                        scaler="MinMax",
                        problem="classification", 
                        models=["lr","dt","rf","mlp"],
                        test_ratio=.20):


    real = real.copy()
    categorical = categorical.copy()

    if target is None:
      target = real.columns[-1]
    
    real, additional_categorical_cols, _=_process_mixed_columns(real, mixed, continuous_placeholder=np.nan)

    categorical.extend(additional_categorical_cols)

    # Encode categorical columns in the real dataset
    for col in categorical:
      label_encoder = LabelEncoder()
      real[col] = label_encoder.fit_transform(real[col])
      
    data_real_y = real.loc[:, target].to_numpy()
    data_real_X = real.drop(target,axis=1).to_numpy()

    data_dim = real.shape[1]
  
    if problem == "classification":
      X_train_real, X_test_real, y_train_real, y_test_real = model_selection.train_test_split(data_real_X ,data_real_y, test_size=test_ratio, stratify=data_real_y,random_state=42) 
    else:
      X_train_real, X_test_real, y_train_real, y_test_real = model_selection.train_test_split(data_real_X ,data_real_y, test_size=test_ratio,random_state=42) 
    
    # Apply scaling
    if scaler=="MinMax":
        scaler_real = MinMaxScaler()
    else:
        scaler_real = StandardScaler()
    

    
    scaler_real.fit(X_train_real)
    X_train_real_scaled = scaler_real.transform(X_train_real)
    X_test_real_scaled = scaler_real.transform(X_test_real)



    if problem == "classification":
      metrics_list = ["Accuracy", "AUC", "F1-Score"]
    else:
      metrics_list = ["MAPE", "Explained Variance", "R2 Score"]


    all_real_results = []
    for model in models:
      real_results = supervised_model_training(X_train_real_scaled,y_train_real,X_test_real_scaled,y_test_real,model,problem)
      all_real_results.append(real_results)
      
    all_fake_results_avg = []
    
    for fake in fakes:
      fake = fake.copy()

      fake, _, _=_process_mixed_columns(fake, mixed, continuous_placeholder=np.nan)

      # Encode categorical columns in the fake dataset
      for col in categorical:
        label_encoder = LabelEncoder()
        fake[col] = label_encoder.fit_transform(fake[col])

      data_fake = fake.to_numpy()

      data_fake_y = fake.loc[:, target].to_numpy()
      data_fake_X = fake.drop(target,axis=1).to_numpy()
     
      
      
      if problem=="classification":
        X_train_fake, _ , y_train_fake, _ = model_selection.train_test_split(data_fake_X ,data_fake_y, test_size=test_ratio, stratify=data_fake_y,random_state=42) 
      else:
        X_train_fake, _ , y_train_fake, _ = model_selection.train_test_split(data_fake_X ,data_fake_y, test_size=test_ratio,random_state=42)  


      # Apply scaling
      if scaler=="MinMax":
        scaler_fake = MinMaxScaler()
      else:
        scaler_fake = StandardScaler()
      
      scaler_fake.fit(data_fake_X)
      
      X_train_fake_scaled = scaler_fake.transform(X_train_fake)
      
      all_fake_results = []
      for model in models:
        fake_results = supervised_model_training(X_train_fake_scaled,y_train_fake,X_test_real_scaled,y_test_real,model,problem)
        all_fake_results.append(fake_results)

      all_fake_results_avg.append(all_fake_results)
    

    

    
    real_results_df = pd.DataFrame(all_real_results, columns=metrics_list, index=models)

    fake_results_dfs = []

    for fake_results in all_fake_results_avg:
      fake_df = pd.DataFrame(fake_results, columns=metrics_list, index=models)
      fake_results_dfs.append(fake_df)

  # Return real and list of fake DataFrames
    return real_results_df, fake_results_dfs




def stat_sim(real,fake,categorical=[],mixed={},mnar=True):
    

    nan_placeholder = "__MISSING__"
    continuous_placeholder='__CONTINUOUS__'


    real = real.copy()
    fake = fake.copy()
    categorical = categorical.copy()

   
    mixed = defaultdict(list, mixed)

    columns_with_nan = []
    if mnar: # If Missing not at random, we replace NaN values with a placeholder,effectively treating them as a separate category
      
      nan_counts_real = real.isna().sum()
      nan_counts_fake = fake.isna().sum()
      columns_with_nan_real = nan_counts_real[nan_counts_real > 0].index.tolist()
      columns_with_nan_fake = nan_counts_fake[nan_counts_fake > 0].index.tolist()

      columns_with_nan = list(set(columns_with_nan_real + columns_with_nan_fake))

      for column in columns_with_nan:
        if continuous_placeholder not in mixed[column]: mixed[column].append(nan_placeholder)

      real = real.fillna(nan_placeholder)
      fake = fake.fillna(nan_placeholder)

    real_processed, additional_categorical_cols, additional_continuous_cols = _process_mixed_columns(real, mixed, continuous_placeholder=continuous_placeholder)
    fake_processed, _, _ = _process_mixed_columns(fake, mixed, continuous_placeholder=continuous_placeholder)
    categorical.extend(additional_categorical_cols)


    corr_diff = alternative_correlation(real_processed, fake_processed, columns_to_remove=additional_continuous_cols) 
    summary, column_stats = column_similarity(real_processed, fake_processed, categorical)

    return summary, column_stats, corr_diff



def column_similarity(real, fake, categorical=[]):


    real = real.copy()
    fake = fake.copy()

    for col in categorical:
      label_encoder = LabelEncoder()
      real[col] = label_encoder.fit_transform(real[col])
      fake[col] = label_encoder.transform(fake[col])


    column_stats = []
    default_columns_weight = 1
    weights = []
    for column in real.columns:
        
        if column in categorical:
            
            real_pdf=(real[column].value_counts()/real[column].value_counts().sum())
            fake_pdf=(fake[column].value_counts()/fake[column].value_counts().sum())
            categories = (fake[column].value_counts()/fake[column].value_counts().sum()).keys().tolist()
            sorted_categories = sorted(categories)
            
            real_pdf_values = [] 
            fake_pdf_values = []

            for i in sorted_categories:
                #if i not in real_pdd: raise ValueError(f"Category {i} present in fake but not in real data")
                if i not in real_pdf: real_pdf[i] = 0 #TODO: might not be like this
                if i not in fake_pdf: fake_pdf[i] = 0
                real_pdf_values.append(real_pdf[i])
                fake_pdf_values.append(fake_pdf[i])
            
                
            js_distance = (distance.jensenshannon(real_pdf_values,fake_pdf_values, 2.0))
 
            statistics = [column, "JSD",js_distance, default_columns_weight]
    
            weights.append(default_columns_weight)
    
        else:
            scaler = MinMaxScaler()
            scaler.fit(real[column].values.reshape(-1,1))
            l1 = scaler.transform(real[column].values.reshape(-1,1)).flatten()
            l2 = scaler.transform(fake[column].values.reshape(-1,1)).flatten()
            weight = default_columns_weight
            #TODO: this might drop stuff it should not
            weight = 1 - np.isnan(l1).sum()/len(l1)
            l1 = l1[~np.isnan(l1)]
            l2 = l2[~np.isnan(l2)]
            w_distance = (wasserstein_distance(l1,l2))
            statistics = [column, "WD", w_distance, weight]
            weights.append(weight)
        column_stats.append(statistics)

    column_stats = pd.DataFrame(column_stats, columns=["Column", "Metric", "Distance", "Weight"])

    summary = column_stats.groupby('Metric').agg({
        'Distance': [
            ('Weighted_Avg', lambda x: np.average(x, weights=column_stats.loc[x.index, 'Weight'])),
            ('Mean', 'mean'),
            ('Std', 'std')
        ]
    })
    summary.columns = summary.columns.get_level_values(1)
    summary = summary.reset_index()
    return summary, column_stats

def alternative_correlation(real, fake, columns_to_remove=[],difference_metric="mae"):
    """
    Calculate the correlation between real and fake datasets
    
    Args:
        real (pd.DataFrame): Real dataset
        fake (pd.DataFrame): Fake dataset
        columns_to_remove (list): List of columns to remove from the datasets
    
    Returns:
        float: Correlation distance
    """
    # Remove specified columns
    real = real.drop(columns=columns_to_remove, errors='ignore')
    fake = fake.drop(columns=columns_to_remove, errors='ignore')

    # Calculate correlation matrices
    real_corr = associations(real, compute_only=True)["corr"]
    fake_corr = associations(fake, compute_only=True)["corr"]

    if difference_metric == "mae":
        # Calculate absolute difference between correlation matrices
        diff_matrix = np.abs(real_corr - fake_corr)
        columnwise_avg_diff = diff_matrix.mean(axis=0)
        return columnwise_avg_diff.mean()
    
    if difference_metric == "forbenious":
        # Calculate Frobenius norm of the difference between correlation matrices
        return np.linalg.norm(real_corr - fake_corr, ord='fro')
    
    raise ValueError(f"Unknown difference metric: {difference_metric}. Supported metrics are 'mae' and 'forbenious'.")
    
    

def _process_mixed_columns(df, mixed, continuous_placeholder='__CONTINUOUS__'):
    """
    Process mixed columns by extracting categorical values
    
    Args:
        df (pd.DataFrame): Input DataFrame
        mixed (dict): Dictionary of mixed columns with their categories
        continuous_placeholder (str): Placeholder to indicate continuous value in categorical part of mixed columns
    
    Returns:
        tuple: (processed_df, additional_categorical_cols)
    """
    # Create a copy of the DataFrame
    df_copy = df.copy()
    additional_categorical_cols = []
    additional_continuous_cols = []
    
    for col, categories in mixed.items():
        # Create a new categorical column
        new_cat_col_name = f"{col}_categorical"
        new_cont_col_name = f"{col}_continuous"
        
        # Create a series for categorical values
        categorical_series = df_copy[col].apply(
            lambda x: x if x in categories else continuous_placeholder
        )
        
        # Add the new categorical column
        df_copy[new_cat_col_name] = categorical_series
        additional_categorical_cols.append(new_cat_col_name)

        continuous_series = df_copy[col].apply(
            lambda x: x if x not in categories else np.nan
        )
        df_copy[new_cont_col_name] = continuous_series
        additional_continuous_cols.append(new_cont_col_name)

        # Drop the original column
        df_copy.drop(columns=[col], inplace=True)
    
    return df_copy, additional_categorical_cols, additional_continuous_cols





def privacy_metrics(real, 
                    fake, 
                    metric = 'gower', 
                    data_percent=15,
                    verbose=False):
    """
    Calculate privacy metrics between real and fake datasets.
    
    Parameters:
    real (DataFrame): Real dataset
    fake (DataFrame): Synthetic dataset
    data_percent (int): Percentage of data to sample for analysis
    
    Returns:
    dict: Dictionary containing detailed privacy metrics
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    import pandas as pd
    
    # Create a metrics dictionary to store all results
    privacy_summary = []
    
    # Sample the data
    real_refined = real.sample(n=int(len(real)*(.01*data_percent)), random_state=42).to_numpy()
    fake_refined = fake.sample(n=int(len(fake)*(.01*data_percent)), random_state=42).to_numpy()
    
    if verbose:
      print("==== Privacy Metrics Analysis ====")
      print(f"Using {data_percent}% of data for analysis")
      
      
      print(f"Real data sample size: {len(real_refined)}")
      print(f"Fake data sample size: {len(fake_refined)}")
      

      print("\nCalculating pairwise distances...")
    
    # Compute distance matrix from real-fake, real-real, and fake-fake
    dist_rf, dist_rr, dist_ff = _calculate_pariwise_distances(real_refined, fake_refined, metric=metric)

    # Remove the diagonal to ignore self-distances
    dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 

    
    
    # Find smallest two distances for each metric
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    smallest_two_indexes_rr = [dist_rr[i].argsort()[:2] for i in range(len(dist_rr))]
    smallest_two_rr = [dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(dist_rr))]
    smallest_two_indexes_ff = [dist_ff[i].argsort()[:2] for i in range(len(dist_ff))]
    smallest_two_ff = [dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(dist_ff))]
    
    # Cacluaate nearest neighbor distance ratios
    nndr_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    nndr_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    nndr_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])

    # Calculate distance to closest record
    dcr_rf = np.array([i[0] for i in smallest_two_rf])
    dcr_rr = np.array([i[0] for i in smallest_two_rr])
    dcr_ff = np.array([i[0] for i in smallest_two_ff])
    
    # We report the lowerst 5th percentile of the metrics
    nndr_fifth_perc_rr = np.percentile(nndr_ratio_rr, 5)
    nndr_fifth_perc_ff = np.percentile(nndr_ratio_ff, 5)
    nndr_fifth_perc_rf = np.percentile(nndr_ratio_rf, 5)
    
    fifth_perc_rf = np.percentile(dcr_rf, 5)
    fifth_perc_rr = np.percentile(dcr_rr, 5)
    fifth_perc_ff = np.percentile(dcr_ff, 5)
    
    

    privacy_percentiles = pd.DataFrame({
    'rr': [nndr_fifth_perc_rr, fifth_perc_rr],
    'ff': [nndr_fifth_perc_ff, fifth_perc_ff],
    'rf': [nndr_fifth_perc_rf, fifth_perc_rf]
    }, index=['nndr_5th', 'dcr_5th'])

    privacy_summary.append(privacy_percentiles)
    
    if verbose:
      print("\n== Nearest Neighbor Ratio Metrics (5th percentile) ==")
      print(f"Real-to-Real NN Ratio (5th): {nndr_fifth_perc_rr:.4f}")
      print(f"Fake-to-Fake NN Ratio (5th): {nndr_fifth_perc_ff:.4f}")
      print(f"Real-to-Fake NN Ratio (5th): {nndr_fifth_perc_rf:.4f}")
    
      print("\n== Minimum Distance Metrics (5th percentile) ==")
      print(f"Real-to-Real Min Distance (5th): {fifth_perc_rr:.4f}")
      print(f"Fake-to-Fake Min Distance (5th): {fifth_perc_ff:.4f}")
      print(f"Real-to-Fake Min Distance (5th): {fifth_perc_rf:.4f}")


    nnaa = compute_nnaa(dist_rf,dist_rr,dist_ff)
    privacy_summary.append(nnaa)

    if verbose:
      print("\n== Nearest Neighbor Adversarial Accuracy (NNAA) ==")
      print(f"NNAA: {nnaa:.4f}")
    
    # Calculate privacy risk score
    # Higher score indicates lower privacy risk
    privacy_risk = (fifth_perc_rf / (fifth_perc_rr + fifth_perc_ff) * 2)
    privacy_summary.append(privacy_risk)
    
    if verbose:
      print(f"\nPrivacy Risk Score: {privacy_risk:.4f}")
      print("(Higher score indicates better privacy protection)")
    
    
    
    
    if verbose:
      print("\n==== Summary Interpretation ====")
      if fifth_perc_rf > (fifth_perc_rr + fifth_perc_ff)/2:
          print("✓ Good distance between real and synthetic records")
      else:
          print("⚠ Synthetic records may be too similar to real data")
          
      if nndr_fifth_perc_rf > (nndr_fifth_perc_rr + nndr_fifth_perc_ff)/2:
          print("✓ Good neighbor distance ratios")
      else:
          print("⚠ Neighbor distance ratios indicate possible privacy concerns")
      
    
    return privacy_summary


# Compute the Nearest Neighbor Adversarial Accuaracy (NNAA) between real and fake data
def compute_nnaa(dist_rf, # Distance matrix between real and fake data
                 dist_rr, # Distance matrix between real and real data
                 dist_ff): # Distance matrix between fake and fake data

    real_data_size = dist_rf.shape[0] # Number of real data points
    fake_data_size = dist_rf.shape[1] # Number of fake data points

    dist_fr = dist_rf.T # Transpose to get distance matrix between fake and real data

    dist_rf_min = np.min(dist_rf, axis=1) # Minimum distance from real to fake
    dist_fr_min = np.min(dist_fr, axis=1) # Minimum distance from fake to real
    dist_rr_min = np.min(dist_rr, axis=1) # Minimum distance from real to real
    dist_ff_min = np.min(dist_ff, axis=1) # Minimum distance from fake to fake

    real_data_closest = (dist_rr_min < dist_rf_min).astype(int) # The closest point to real data is reals data
    fake_data_closest = (dist_ff_min < dist_fr_min).astype(int) # The closest point to fake data is fakes data

    nnaa = (real_data_closest.mean() + fake_data_closest.mean()) / 2 # Take the mean of both and average them

    return nnaa



def _calculate_pariwise_distances(real, fake, metric='gower'):
    dist_rf = 0 # Pairwise distance between real and fake
    dist_rr = 0 # Pairwise distance in real data
    dist_ff = 0 # Pairwise distance between fake and fake
    if metric == 'gower':
      dist_rf = gower_distance(real,fake)
      dist_rr = gower_distance(real,real)
      dist_ff = gower_distance(fake,fake)
      return dist_rf, dist_rr, dist_ff


    if metric == 'euclidean':
      scalerR = StandardScaler()
      scalerF = StandardScaler()
      scalerR.fit(real)
      scalerF.fit(fake)
      df_real_scaled = scalerR.transform(real)
      df_fake_scaled = scalerF.transform(fake)

      dist_rf = metrics.pairwise_distances(df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)
      dist_rr = metrics.pairwise_distances(df_real_scaled, Y=None, metric='minkowski', n_jobs=-1)
      dist_ff = metrics.pairwise_distances(df_fake_scaled, Y=None, metric='minkowski', n_jobs=-1)

      return dist_rf, dist_rr, dist_ff
    

    raise ValueError(f"Unknown metric: {metric}. Supported metrics are 'gower' and 'euclidean'.")

