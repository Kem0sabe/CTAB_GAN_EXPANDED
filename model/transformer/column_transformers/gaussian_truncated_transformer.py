import numpy as np
import torch
from scipy.stats import norm
from scipy.optimize import minimize
from .column_transformer import Column_transformer


class Gaussian_truncated_transformer(Column_transformer):
    def __init__(self, column):
        super().__init__()
        self.min = column.min()
        self.max = column.max()

        self.fit(column.to_numpy())

    def fit(self, data_col):
        self.model = None 
        self.components = None
        self.output_info = [(1, 'tanh','yes_g')]
        self.output_dim = 1

    def transform(self, data_col):
        
        self.mu, self.sigma = fit_truncated_normal(data_col, self.min, self.max)
        transformed = (data_col - self.mu) / self.sigma
        self.transformed_min = transformed.min()*1.02
        self.transformed_max = transformed.max()*1.02
        transformed_min_max = (transformed - self.transformed_min) / (self.transformed_max -self.transformed_min) * 2 - 1
        return transformed_min_max.reshape([-1, 1])
        

    def inverse_transform(self, data,st):
        transformed_min_max = data[:, st]
        transformed_min_max = np.clip(transformed_min_max, -1, 1)
        transformed = (transformed_min_max + 1) / 2 * (self.transformed_max - self.transformed_min) + self.transformed_min
        u = transformed * self.sigma + self.mu
        new_st = st + 1
        return u, new_st, []


    def inverse_transform_static(self, data, transformer, st, device, n_clusters=10): #TODO: remove n cluster when finding out how to handle it
        u = data[:, st]
        u = (u + 1) / 2
        u = torch.clamp(u, 0, 1)
        u = u * (transformer.max - transformer.min) + transformer.min
        new_st = st + 1
        return u, new_st, []



def fit_truncated_normal(data, lower_cutoff = None, upper_cutoff = None):
    if lower_cutoff is None:
        lower_cutoff = data.min()
    if upper_cutoff is None:
        upper_cutoff = data.max()
    
    truncated_data = data[(data >= lower_cutoff) & (data <= upper_cutoff)]
    init_guess = [np.mean(truncated_data), np.std(truncated_data)]
    result = minimize(
        neg_log_likelihood_two_sided,
        x0=init_guess,
        args=(truncated_data, lower_cutoff, upper_cutoff),
        bounds=[(None, None), (1e-5, None)],
        method='L-BFGS-B'
    )
    mu, sigma = result.x
    return mu, sigma


def neg_log_likelihood_two_sided(params, data, a, b):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    z = (data - mu) / sigma
    alpha = (a - mu) / sigma
    beta = (b - mu) / sigma
    logpdf = norm.logpdf(z)
    trunc_logprob = np.log(norm.cdf(beta) - norm.cdf(alpha))
    ll = np.sum(logpdf) - len(data) * np.log(sigma) - len(data) * trunc_logprob
    return -ll
        

        
        

