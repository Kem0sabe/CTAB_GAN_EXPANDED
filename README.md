# CTAB_XTRA_DP
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/Kem0sabe/CTAB_XTRA_DP) 
[![PyPI version](https://badge.fury.io/py/ctab-xtra-dp.svg)](https://badge.fury.io/py/ctab-xtra-dp)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


A privacy-preserving synthetic tabular data generator based on GANs.

## Installation

```bash
pip install --upgrade ctab-xtra-dp
```

## Overview

`CTAB_XTRA_DP` is a generative model for creating high-quality synthetic tabular data with differential privacy guarantees. It extends the [CTAB-GAN+](https://github.com/Team-TUD/CTAB-GAN-Plus) architecture to generate synthetic datasets that preserve the statistical properties of the original data while providing formal privacy protection.

## Features

- Generate synthetic tabular data with similar statistical properties to the original data
- Automatic handling of various data types (categorical, numerical, mixed, log-transformed)
- Handles Missing Not at Random (MNAR) null values
- Built-in differential privacy to protect sensitive information
- Conditional generation based on specific column values

## Quick Start

```python
from ctab_xtra_dp import CTAB_XTRA_DP , load_demo
import pandas as pd

# Load your data
df = load_demo("car).drop(columns=['Year','Model'])

# Initialize the model
synthesizer = CTAB_XTRA_DP(
    df=df,
    categorical_columns=["Brand","Fuel_Type","Transmission"],
)

# Train the model
synthesizer.fit(epochs=50)

# Generate synthetic data
synthetic_data = synthesizer.generate_samples(n=df.shape[0])



# Generate samples with specific conditions
synthetic_data_electric = synthesizer.generate_samples(
    n=500, 
    conditioning_column='Fuel_Type', 
    conditioning_value='Electric'
)
```

## API Reference

### CTAB_XTRA_DP

```python
CTAB_XTRA_DP(
    df,
    categorical_columns=[], 
    log_columns=[],
    mixed_columns={},
    gaussian_columns=[],
    integer_columns=[],
    problem_type=("Classification", 'target_column'),
    dp_constraints={
        "epsilon_budget": 10,
        "delta": None,
        "sigma": None,
        "clip_coeff": 1
    }
)
```

#### Parameters

- **df** : pandas.DataFrame
  - The input dataframe to train on
  
- **categorical_columns** : list
  - List of column names that should be treated as categorical
  
- **log_columns** : list
  - List of column names that should be log-transformed before modeling
  
- **mixed_columns** : dict
  - Dictionary mapping column names to their unique modal values
  - Used for columns with mixed continuous-discrete distributions
  - Example: `{'capital-loss': [0]}` indicates that 'capital-loss' has a special value at 0
  - Specifying `{'capital-loss': [np.null]}` indicates to the model that we have a MNAR value
  
- **gaussian_columns** : list
  - List of column names that should be modeled with a Gaussian distribution
  
- **integer_columns** : list
  - List of column names that should be treated as integers
  - This overwrites the original datatype purposed to the model
  - If the column type is interger, this overwrite is not neccesary
  
- **problem_type** : tuple
  - Set the target column used for the auxiliary classifier during training
  - A tuple of (problem_type, target_column)
  - problem_type can be "Classification" or "Regression"
  - If sett to None, no auxiliary classifier is used. (The generation works more then fine without it)
    
  
- **dp_constraints** : dict
  - Differential privacy parameters:
    - **epsilon_budget**: Privacy budget for the entire training process. Computes the sigma noise for the given epsilon to ensure privacy guarentees.
    - **delta**: Probabilistic relaxation parameter, should be set to a number much less than 1/n (default: 0.1/n)
    - **sigma**: Gaussian noise to be added for each itteration. If this is set, it overrides any epsilon value.
    - **clip_coeff**: Coefficient for gradient clipping. A common practice is to leave this at 1. (default: 1)

### Methods

#### fit

```python
fit(epochs=100)
```

Train the model on the input dataframe provided in the constructor.

**Parameters:**
- **epochs** : int
  - Number of training epochs (default: 100)

**Returns:**
- None

#### generate_samples

```python
generate_samples(n=100, conditioning_column=None, conditioning_value=None)
```

Generate synthetic samples with option to conditional generation.

**Parameters:**
- **n** : int
  - Number of synthetic samples to generate (default: 100)
- **conditioning_column** : str, optional
  - Column name to condition on
- **conditioning_value** : any, optional
  - Value of the conditioning column

**Returns:**
- **pandas.DataFrame**
  - Generated synthetic data

## Data Type Handling

`CTAB_XTRA_DP` automatically processes different data types: **This is not yet implemented**

- **Categorical data**: One-hot encoded
- **Mixed data**: Modeled using a mixture of discrete modes and continuous distributions
- **Log-transformed data**: Log-transformed before modeling, exponentiated during generation
- **Integer data**: Values are rounded to integers during generation
- **Gaussian data**: Modeled directly with a Gaussian distribution

## Differential Privacy

The model implements differential privacy using:

- Gradient clipping to bound the sensitivity of the training process
- Gaussian noise addition to gradients according to the specified privacy parameters
- Privacy accounting to track epsilon expenditure

## Examples

### Basic Usage

```python
import pandas as pd
from ctab_xtra_dp import CTAB_XTRA_DP, load_demo

# Load data
df = load_demo()

# Initialize model
synthesizer = CTAB_XTRA_DP(
    df=df,
    categorical_columns=['workclass', 'education', 'marital-status', 'occupation', 
                         'relationship', 'race', 'gender', 'native-country'],
    mixed_columns={'capital-loss': [0], 'capital-gain': [0]},
    integer_columns=['age', 'fnlwgt', 'hours-per-week']
)

# Train model
synthesizer.fit(epochs=150)

# Generate synthetic data
synthetic_data = synthesizer.generate_samples(n=1000)

# Save synthetic data
synthetic_data.to_csv("synthetic_adult.csv", index=False)
```

### Generating Conditioned Samples

```python
# Generate samples with specific education level
bachelors_samples = synthesizer.generate_samples(
    n=500,
    conditioning_column='education',
    conditioning_value='Bachelors'
)

# Generate samples with specific occupation
managers_samples = synthesizer.generate_samples(
    n=500,
    conditioning_column='occupation',
    conditioning_value='Exec-managerial'
)
```
### With auxiliary classifier
```python
import pandas as pd
from ctab_xtra_dp import CTAB_XTRA_DP, load_demo

# Load data
df = load_demo()

# Initialize model
synthesizer = CTAB_XTRA_DP(
    df=df,
    categorical_columns=['workclass', 'education', 'marital-status', 'occupation', 
                         'relationship', 'race', 'gender', 'native-country',
    mixed_columns={'capital-loss': [0], 'capital-gain': [0]},
    integer_columns=['age', 'fnlwgt', 'hours-per-week'],
    problem_type=("Classification", 'income')
)

# Train model
synthesizer.fit(epochs=150)

# Generate synthetic data
synthetic_data = synthesizer.generate_samples(n=1000)

# Save synthetic data
synthetic_data.to_csv("synthetic_adult.csv", index=False)
```


### With Differential Privacy

```python
# Initialize with stronger privacy guarantees
private_synthesizer = CTAB_XTRA_DP(
    df=df,
    categorical_columns=['workclass', 'education', 'marital-status', 'occupation', 
                         'relationship', 'race', 'gender', 'native-country'],
    integer_columns=['age', 'hours-per-week'],
    dp_constraints={
        "epsilon_budget": 1.0,  # Stricter privacy budget
        "clip_coeff": 1.0
    }
)

private_synthesizer.fit(epochs=100)
private_samples = private_synthesizer.generate_samples(n=1000)
```
Not specifying delta allows the model to compute a resonable delta value.

### With MNAR value
A handfull of the 'capital-loss' and 'capital-gain' has missing financial data. In this case removing null values would loose valueable information. At the same time, interperating it as 0 will not distinguios does who do not have much financial activity from people who have null in the system some access reason. 
```python
# Initialize with stronger privacy guarantees
private_synthesizer = CTAB_XTRA_DP(
    df=df,
    categorical_columns=['workclass', 'education', 'marital-status', 'occupation', 
                         'relationship', 'race', 'gender', 'native-country'],
    integer_columns=['age', 'hours-per-week'],
    mixed_columns={'capital-loss': [0,np.nan], 'capital-gain': [0,np.nan]},
    dp_constraints={
        "epsilon_budget": 1.0,  # Stricter privacy budget
        "clip_coeff": 1.0
    }
)

private_synthesizer.fit(epochs=100)
private_samples = private_synthesizer.generate_samples(n=1000)
```
Not specifying delta allows the model to compute a resonable delta value.
## Evaluation Framework

CTAB_XTRA_DP includes a comprehensive evaluation framework to assess both the utility and privacy of the generated synthetic data.

### Utility Evaluation

```python
from ctab_xtra_dp.evaluation import get_utility_metrics, stat_sim

# Load synthetic data

synthetic_data = synthesizer.generate_samples(n=1000)

# Evaluate supervised learning performance difference
# Lower difference values indicate better utility preservation
utility_diff = get_utility_metrics(
    data_real=df,
    data_synthetic=synthetic_data,
    scaler="MinMax",  # or "Standard"
    type={"Classification": ["lr", "dt", "rf", "mlp"]},  # for classification tasks
    # type={"Regression": ["l_reg", "ridge", "lasso", "B_ridge"]},  # for regression tasks
    test_ratio=0.2
)

# Evaluate statistical similarity
# Lower values indicate better statistical preservation
cat_columns = ['workclass', 'education', 'marital-status', 'occupation']
stat_metrics = stat_sim(real_data, synthetic_data, cat_cols=cat_columns)

# stat_metrics[0]: Average Wasserstein distance for numerical columns
# stat_metrics[1]: Average Jensen-Shannon divergence for categorical columns
# stat_metrics[2]: Correlation matrix distance
```

### Privacy Evaluation

```python
from ctab_xtra_dp.evaluation import privacy_metrics

# Assess privacy protection
privacy_results = privacy_metrics(
    real=df, 
    fake=synthetic_data, 
    data_percent=15  # Percentage of data to sample for efficiency
)

# Key metrics in privacy_results:
# - min_dist_rf_5th: Minimum distance from real to fake records (5th percentile)
# - min_dist_rr_5th: Minimum distance within real records (5th percentile)
# - min_dist_ff_5th: Minimum distance within fake records (5th percentile)
# - privacy_risk_score: Overall privacy protection score (higher is better)
```

### Interpreting Evaluation Results

#### Utility Metrics
- **Machine Learning Performance**: Lower difference values indicate synthetic data that better preserves predictive relationships
- **Statistical Similarity**: Lower values indicate better preservation of distributions and correlations

#### Privacy Metrics
- **Minimum Distances**: Higher real-to-fake distances relative to within-dataset distances suggest better privacy protection
- **Privacy Risk Score**: Values greater than 1.0 indicate good privacy protection, with higher values being better


### Complete Evaluation Example

```python
import pandas as pd
from ctab_xtra_dp import CTAB_XTRA_DP, load_demo
from ctab_xtra_dp.evaluation import get_utility_metrics, stat_sim, privacy_metrics

# Load data
real_data = load_demo()

# Initialize and train model
synthesizer = CTAB_XTRA_DP(
    df=real_data,
    categorical_columns=['workclass', 'education', 'marital-status', 'occupation',
                        'relationship', 'race', 'gender', 'native-country'],
    mixed_columns={'capital-loss': [0], 'capital-gain': [0]},
    integer_columns=['age', 'hours-per-week'],
    dp_constraints={"epsilon_budget": 5.0}
)

synthesizer.fit(epochs=100)
synthetic_data = synthesizer.generate_samples(n=len(real_data))

# Comprehensive evaluation
# 1. Utility evaluation
cat_cols = ['workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'gender', 'native-country', 'income']

ml_diff = get_utility_metrics(
    data_real=real_data,
    data_synthetic=synthetic_data,
    type={"Classification": ["lr", "dt", "rf"]},
    test_ratio=0.2
)

print("Machine Learning Utility Difference:")
print(f"Accuracy diff: {ml_diff[0][0]:.4f}")
print(f"AUC diff: {ml_diff[0][1]:.4f}")
print(f"F1-score diff: {ml_diff[0][2]:.4f}")

# 2. Statistical similarity
stat_results = stat_sim(real_data, synthetic_data, cat_cols=cat_cols)
print("\nStatistical Similarity:")
print(f"Numerical columns (Wasserstein): {stat_results[0]:.4f}")
print(f"Categorical columns (JSD): {stat_results[1]:.4f}")
print(f"Correlation distance: {stat_results[2]:.4f}")

# 3. Privacy evaluation
priv_results = privacy_metrics(real_data, synthetic_data)
print("\nPrivacy Evaluation:")
print(f"Privacy Risk Score: {priv_results['privacy_risk_score']:.4f}")
print(f"Real-to-Fake Min Distance (5th): {priv_results['min_dist_rf_5th']:.4f}")
print(f"Real-to-Real Min Distance (5th): {priv_results['min_dist_rr_5th']:.4f}")
```




## Citation
**The citation is not yet avaliable**
If you use this package in your research, please cite:

```bibtex
@article{ctab-xtra-dp,
  title={CTAB-XTRA-DP: Improved Tabular Data Synthesis with Differential Privacy},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```


## License

MIT
