import numpy as np
import pandas as pd 
from sklearn import metrics
from sklearn import model_selection
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, BayesianRidge
from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial import distance
import warnings
from .gower_mix import gower_distance
from collections import defaultdict

warnings.filterwarnings("ignore")

def supervised_model_training(x_train, y_train, x_test, 
                              y_test, model_name,problem_type):
  
  if model_name == 'lr':
    model  = LogisticRegression(random_state=42,max_iter=500) 
  elif model_name == 'svm':
    model  = svm.SVC(random_state=42,probability=True)
  elif model_name == 'dt':
    model  = tree.DecisionTreeClassifier(random_state=42)
  elif model_name == 'rf':      
    model = RandomForestClassifier(random_state=42)
  elif model_name == "mlp":
    model = MLPClassifier(random_state=42,max_iter=100)
  elif model_name == "l_reg":
    model = LinearRegression()
  elif model_name == "ridge":
    model = Ridge(random_state=42)
  elif model_name == "lasso":
    model = Lasso(random_state=42)
  elif model_name == "B_ridge":
    model = BayesianRidge()
  
  model.fit(x_train, y_train)
  pred = model.predict(x_test)

  if problem_type == "Classification":
    if len(np.unique(y_train))>2:
      predict = model.predict_proba(x_test)        
      acc = metrics.accuracy_score(y_test,pred)*100
      auc = metrics.roc_auc_score(y_test, predict,average="weighted",multi_class="ovr")
      f1_score = metrics.precision_recall_fscore_support(y_test, pred,average="weighted")[2]
      return [acc, auc, f1_score] 

    else:
      predict = model.predict_proba(x_test)[:,1]    
      acc = metrics.accuracy_score(y_test,pred)*100
      auc = metrics.roc_auc_score(y_test, predict)
      f1_score = metrics.precision_recall_fscore_support(y_test,pred)[2].mean()
      return [acc, auc, f1_score] 
  
  else:
    mse = metrics.mean_absolute_percentage_error(y_test,pred)
    evs = metrics.explained_variance_score(y_test, pred)
    r2_score = metrics.r2_score(y_test,pred)
    return [mse, evs, r2_score]


#TODO: Fix to work with pandas df instead of path
def get_utility_metrics(data_real,fake_paths,scaler="MinMax",type={"Classification":["lr","dt","rf","mlp"]},test_ratio=.20):

    data_real = pd.read_csv(real_path).to_numpy()
    data_dim = data_real.shape[1]

    data_real_y = data_real[:,-1]
    data_real_X = data_real[:,:data_dim-1]

    problem = list(type.keys())[0]
    
    models = list(type.values())[0]
    
    if problem == "Classification":
      X_train_real, X_test_real, y_train_real, y_test_real = model_selection.train_test_split(data_real_X ,data_real_y, test_size=test_ratio, stratify=data_real_y,random_state=42) 
    else:
      X_train_real, X_test_real, y_train_real, y_test_real = model_selection.train_test_split(data_real_X ,data_real_y, test_size=test_ratio,random_state=42) 
    

    if scaler=="MinMax":
        scaler_real = MinMaxScaler()
    else:
        scaler_real = StandardScaler()
        
    scaler_real.fit(X_train_real)
    X_train_real_scaled = scaler_real.transform(X_train_real)
    X_test_real_scaled = scaler_real.transform(X_test_real)

    all_real_results = []
    for model in models:
      real_results = supervised_model_training(X_train_real_scaled,y_train_real,X_test_real_scaled,y_test_real,model,problem)
      all_real_results.append(real_results)
      
    all_fake_results_avg = []
    
    for fake_path in fake_paths:
      data_fake  = pd.read_csv(fake_path).to_numpy()
      data_fake_y = data_fake[:,-1]
      data_fake_X = data_fake[:,:data_dim-1]

      if problem=="Classification":
        X_train_fake, _ , y_train_fake, _ = model_selection.train_test_split(data_fake_X ,data_fake_y, test_size=test_ratio, stratify=data_fake_y,random_state=42) 
      else:
        X_train_fake, _ , y_train_fake, _ = model_selection.train_test_split(data_fake_X ,data_fake_y, test_size=test_ratio,random_state=42)  

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
    
    diff_results = np.array(all_real_results)- np.array(all_fake_results_avg).mean(axis=0)

    return diff_results

def stat_sim(real,fake,categorical=[]):
    
    Stat_dict={}
    
    #real = pd.read_csv(real_path)
    #fake = pd.read_csv(fake_path)

    really = real.copy()
    fakey = fake.copy()

    real_corr = associations(real, compute_only=True)["corr"]

    fake_corr = associations(fake, compute_only=True)["corr"]

    corr_dist = np.linalg.norm(real_corr - fake_corr)
    
    cat_stat = []
    num_stat = []
    
    for column in real.columns:
        
        if column in categorical:

            real_pdf=(really[column].value_counts()/really[column].value_counts().sum())
            fake_pdf=(fakey[column].value_counts()/fakey[column].value_counts().sum())
            categories = (fakey[column].value_counts()/fakey[column].value_counts().sum()).keys().tolist()
            sorted_categories = sorted(categories)
            
            real_pdf_values = [] 
            fake_pdf_values = []

            for i in sorted_categories:
                real_pdf_values.append(real_pdf[i])
                fake_pdf_values.append(fake_pdf[i])
            
            if len(real_pdf)!=len(fake_pdf):
                zero_cats = set(really[column].value_counts().keys())-set(fakey[column].value_counts().keys())
                for z in zero_cats:
                    real_pdf_values.append(real_pdf[z])
                    fake_pdf_values.append(0)
            Stat_dict[column]=(distance.jensenshannon(real_pdf_values,fake_pdf_values, 2.0))
            cat_stat.append(Stat_dict[column])    
            print("column: ", column, "JSD: ", Stat_dict[column])  
        else:
            scaler = MinMaxScaler()
            scaler.fit(real[column].values.reshape(-1,1))
            l1 = scaler.transform(real[column].values.reshape(-1,1)).flatten()
            l2 = scaler.transform(fake[column].values.reshape(-1,1)).flatten()
            Stat_dict[column]= (wasserstein_distance(l1,l2))
            print("column: ", column, "WD: ", Stat_dict[column])
            num_stat.append(Stat_dict[column])

    return [np.mean(num_stat),np.mean(cat_stat),corr_dist]


def stat_sim2(real,fake,categorical=[],mixed={},mnar=True):
    

    nan_placeholder = "__MISSING__"
    continuous_placeholder='__CONTINUOUS__'



    #real = pd.read_csv(real_path)
    #fake = pd.read_csv(fake_path)

    really = real.copy()
    fakey = fake.copy()

    categorical = set(categorical)
    mixed = defaultdict(list, mixed)

    columns_with_nan = []
    if mnar: # If Missing not at random, we replace NaN values with a placeholder,effectively treating them as a separate category
      # Replace NaN values with a placeholder
      nan_counts_real = really.isna().sum()
      nan_counts_fake = fakey.isna().sum()
      columns_with_nan_real = nan_counts_real[nan_counts_real > 0].index.tolist()
      columns_with_nan_fake = nan_counts_fake[nan_counts_fake > 0].index.tolist()

      columns_with_nan = list(set(columns_with_nan_real + columns_with_nan_fake))

      for column in columns_with_nan:
        if continuous_placeholder not in mixed[column]: mixed[column].append(nan_placeholder)

      really = really.fillna(nan_placeholder)
      fakey = fakey.fillna(nan_placeholder)

    real_processed, additional_categorical_cols = _process_mixed_columns(really, mixed, continuous_placeholder=continuous_placeholder)
    fake_processed, _ = _process_mixed_columns(fakey, mixed, continuous_placeholder=continuous_placeholder)

    categorical.update(additional_categorical_cols)


        

    
    
  
    column_stats = []
    
    default_columns_weight = 1
    weights = []
    for column in real_processed.columns:
        
        if column in categorical:
            
            real_pdf=(real_processed[column].value_counts()/real_processed[column].value_counts().sum())
            fake_pdf=(fake_processed[column].value_counts()/fake_processed[column].value_counts().sum())
            categories = (fake_processed[column].value_counts()/fake_processed[column].value_counts().sum()).keys().tolist()
            sorted_categories = sorted(categories)
            
            real_pdf_values = [] 
            fake_pdf_values = []

            for i in sorted_categories:
                #if i not in real_pdd: raise ValueError(f"Category {i} present in fake but not in real data")
                if i not in real_pdf: real_pdf[i] = 0 #TODO: might not be like this
                if i not in fake_pdf: fake_pdf[i] = 0
                real_pdf_values.append(real_pdf[i])
                fake_pdf_values.append(fake_pdf[i])
            
            if len(real_pdf)!=len(fake_pdf):
                zero_cats = set(really[column].value_counts().keys())-set(fakey[column].value_counts().keys())
                
            js_distance = (distance.jensenshannon(real_pdf_values,fake_pdf_values, 2.0))
 
            statistics = [column, "JSD",js_distance, default_columns_weight]
    
            weights.append(default_columns_weight)
    
        else:
            scaler = MinMaxScaler()
            scaler.fit(real_processed[column].values.reshape(-1,1))
            l1 = scaler.transform(real_processed[column].values.reshape(-1,1)).flatten()
            l2 = scaler.transform(fake_processed[column].values.reshape(-1,1)).flatten()
            weight = default_columns_weight
            if mnar: # If missing at random the np.nan are just placeholder so we remove them
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


    """
    real_corr = associations(real_processed, compute_only=True)["corr"]
    fake_corr = associations(fake_processed, compute_only=True)["corr"]
    weighted_correlation = (real_corr - fake_corr) * weights

    corr_dist = np.linalg.norm(weighted_correlation)
    """
    
    return summary, column_stats


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

        # Drop the original column
        df_copy.drop(columns=[col], inplace=True)
    
    return df_copy, additional_categorical_cols





def privacy_metrics(real, fake, metric = 'gower', data_percent=15):
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
    privacy_dict = {}
    
    print("==== Privacy Metrics Analysis ====")
    print(f"Using {data_percent}% of data for analysis")
    
    # Sample the data
    real_refined = real.sample(n=int(len(real)*(.01*data_percent)), random_state=42).to_numpy()
    fake_refined = fake.sample(n=int(len(fake)*(.01*data_percent)), random_state=42).to_numpy()
    
    print(f"Real data sample size: {len(real_refined)}")
    print(f"Fake data sample size: {len(fake_refined)}")
    

    print("\nCalculating pairwise distances...")
    
    dist_rf, dist_rr, dist_ff = _calculate_pariwise_distances(real_refined, fake_refined, metric=metric)
    # Remove the distance between the same records (matrix diagonal)
    rd_dist_rr = dist_rr[~np.eye(dist_rr.shape[0],dtype=bool)].reshape(dist_rr.shape[0],-1)
    rd_dist_ff = dist_ff[~np.eye(dist_ff.shape[0],dtype=bool)].reshape(dist_ff.shape[0],-1) 

    
    
    # Find smallest two distances for each metric
    smallest_two_indexes_rf = [dist_rf[i].argsort()[:2] for i in range(len(dist_rf))]
    smallest_two_rf = [dist_rf[i][smallest_two_indexes_rf[i]] for i in range(len(dist_rf))]       
    smallest_two_indexes_rr = [rd_dist_rr[i].argsort()[:2] for i in range(len(rd_dist_rr))]
    smallest_two_rr = [rd_dist_rr[i][smallest_two_indexes_rr[i]] for i in range(len(rd_dist_rr))]
    smallest_two_indexes_ff = [rd_dist_ff[i].argsort()[:2] for i in range(len(rd_dist_ff))]
    smallest_two_ff = [rd_dist_ff[i][smallest_two_indexes_ff[i]] for i in range(len(rd_dist_ff))]
    
    # Calculate nearest neighbor ratios
    nn_ratio_rr = np.array([i[0]/i[1] for i in smallest_two_rr])
    nn_ratio_ff = np.array([i[0]/i[1] for i in smallest_two_ff])
    nn_ratio_rf = np.array([i[0]/i[1] for i in smallest_two_rf])
    
    # Calculate 5th percentiles of ratios
    nn_fifth_perc_rr = np.percentile(nn_ratio_rr, 5)
    nn_fifth_perc_ff = np.percentile(nn_ratio_ff, 5)
    nn_fifth_perc_rf = np.percentile(nn_ratio_rf, 5)
    
    # Store NN ratio metrics
    privacy_dict['nn_ratio_rr_5th'] = nn_fifth_perc_rr
    privacy_dict['nn_ratio_ff_5th'] = nn_fifth_perc_ff
    privacy_dict['nn_ratio_rf_5th'] = nn_fifth_perc_rf
    
    print("\n== Nearest Neighbor Ratio Metrics (5th percentile) ==")
    print(f"Real-to-Real NN Ratio (5th): {nn_fifth_perc_rr:.4f}")
    print(f"Fake-to-Fake NN Ratio (5th): {nn_fifth_perc_ff:.4f}")
    print(f"Real-to-Fake NN Ratio (5th): {nn_fifth_perc_rf:.4f}")
    
    # Calculate minimum distances
    min_dist_rf = np.array([i[0] for i in smallest_two_rf])
    min_dist_rr = np.array([i[0] for i in smallest_two_rr])
    min_dist_ff = np.array([i[0] for i in smallest_two_ff])
    
    # Calculate 5th percentiles of minimum distances
    fifth_perc_rf = np.percentile(min_dist_rf, 5)
    fifth_perc_rr = np.percentile(min_dist_rr, 5)
    fifth_perc_ff = np.percentile(min_dist_ff, 5)
    
    # Store minimum distance metrics
    privacy_dict['min_dist_rf_5th'] = fifth_perc_rf
    privacy_dict['min_dist_rr_5th'] = fifth_perc_rr
    privacy_dict['min_dist_ff_5th'] = fifth_perc_ff
    
    print("\n== Minimum Distance Metrics (5th percentile) ==")
    print(f"Real-to-Fake Min Distance (5th): {fifth_perc_rf:.4f}")
    print(f"Real-to-Real Min Distance (5th): {fifth_perc_rr:.4f}")
    print(f"Fake-to-Fake Min Distance (5th): {fifth_perc_ff:.4f}")
    
    # Calculate privacy risk score
    # Higher score indicates lower privacy risk
    privacy_risk = (fifth_perc_rf / (fifth_perc_rr + fifth_perc_ff) * 2)
    privacy_dict['privacy_risk_score'] = privacy_risk
    
    print(f"\nPrivacy Risk Score: {privacy_risk:.4f}")
    print("(Higher score indicates better privacy protection)")
    
    # Create the final metrics array as before
    metrics_array = np.array([
        fifth_perc_rf, fifth_perc_rr, fifth_perc_ff,
        nn_fifth_perc_rf, nn_fifth_perc_rr, nn_fifth_perc_ff
    ]).reshape(1, 6)
    
    privacy_dict['metrics_array'] = metrics_array
    
    print("\n==== Summary Interpretation ====")
    if fifth_perc_rf > (fifth_perc_rr + fifth_perc_ff)/2:
        print("✓ Good distance between real and synthetic records")
    else:
        print("⚠ Synthetic records may be too similar to real data")
        
    if nn_fifth_perc_rf > (nn_fifth_perc_rr + nn_fifth_perc_ff)/2:
        print("✓ Good neighbor distance ratios")
    else:
        print("⚠ Neighbor distance ratios indicate possible privacy concerns")
    
    return privacy_dict




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

