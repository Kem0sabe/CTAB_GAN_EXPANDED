import numpy as np
import pandas as pd
import torch
from sklearn.mixture import BayesianGaussianMixture
from .column_transformer import Column_transformer
class GMM_transformer(Column_transformer):
    def __init__(self, column,n_clusters=10,eps=0.005):
        super().__init__()
        self.n_clusters = n_clusters
        self.eps = eps
        self.min = column.min()
        self.max = column.max()
        
        self.fit(column.to_numpy())


    def fit(self, data_col):
        gm = BayesianGaussianMixture(
                n_components = self.n_clusters, 
                weight_concentration_prior_type='dirichlet_process',
                weight_concentration_prior=0.001, 
                max_iter=100,n_init=1, random_state=42)
        gm.fit(data_col.reshape([-1, 1]))
        mode_freq = (pd.Series(gm.predict(data_col.reshape([-1, 1]))).value_counts().keys())
        
        old_comp = gm.weights_ > self.eps

        # The final components are the ones that are frequent and have a weigh greater than the eps cutoff
        self.components = [(i in mode_freq) & old_comp[i] for i in range(self.n_clusters)]
        self.output_info = [(1, 'tanh','no_g'), (np.sum(self.components), 'softmax',None)]
        self.output_dim = 1 + np.sum(self.components)
        self.model = gm
        

    def transform(self, data_col):
        np.random.seed(22) # TODO: remove used for debugging
        data_col = data_col.reshape([-1, 1])
        means = self.model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(self.model.covariances_).reshape((1, self.n_clusters))
        features = np.empty(shape=(len(data_col),self.n_clusters))
       
        features = (data_col - means) / (4 * stds)

        probs = self.model.predict_proba(data_col.reshape([-1, 1]))
        n_opts = sum(self.components)
        features = features[:, self.components]
        probs = probs[:, self.components]

        opt_sel = np.zeros(len(data_col), dtype='int')
        for i in range(len(data_col)):
            pp = probs[i] + 1e-6
            pp = pp / sum(pp)
            opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)

        idx = np.arange((len(features)))
        features = features[idx, opt_sel].reshape([-1, 1])
        features = np.clip(features, -.99, .99) 
        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1

        re_ordered_phot = np.zeros_like(probs_onehot)
        
        col_sums = probs_onehot.sum(axis=0)
        

        n = probs_onehot.shape[1]
        largest_indices = np.argsort(-1*col_sums)[:n]
        self.ordering = largest_indices
        for id,val in enumerate(largest_indices):
            re_ordered_phot[:,id] = probs_onehot[:,val]
    
        
        return np.concatenate([features, re_ordered_phot], axis=1)
        

    def inverse_transform(self, data,st):
        u = data[:, st]
        v = data[:, st + 1:st + 1 + np.sum(self.components)]
      
        v_re_ordered = np.zeros_like(v)

        for id,val in enumerate(self.ordering):
            v_re_ordered[:,val] = v[:,id]
        
        v = v_re_ordered

        u = np.clip(u, -1, 1)
        v_t = np.ones((data.shape[0], self.n_clusters)) * -100
        v_t[:, self.components] = v
        v = v_t
        
        means = self.model.means_.reshape([-1])
        stds = np.sqrt(self.model.covariances_).reshape([-1])
        p_argmax = np.argmax(v, axis=1)
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]
        tmp = u * 4 * std_t + mean_t
        
        new_st = st + 1 + np.sum(self.components)
        return tmp, new_st, []

    def inverse_transform_static(self, data, transformer,st,device,n_clusters=10):
        components = transformer.get_component()
        order = transformer.get_ordering()
        model = transformer.get_model()

        components = torch.tensor(components, dtype=torch.bool).to(device)
        u = data[:, st]  # Then tahn component for "concatenating" the different components
        v = data[:, st + 1:st + 1 + torch.sum(components).item()]  # The different components
        

        # Ensure order is a tensor
        order = torch.tensor(order, device=device)
        v = v[:, torch.argsort(order)]
        
        v_t = torch.ones((data.shape[0], n_clusters), device=device) * -float('inf')
        v_t[:, components] = v
        v = v_t

        p_argmax = torch.argmax(v, dim=1)
        means = torch.tensor(model.means_.reshape([-1]), device=device)
        stds = torch.tensor(model.covariances_.reshape([-1]), device=device)
        stds = torch.sqrt(stds)    
        std_t = stds[p_argmax]
        mean_t = means[p_argmax]

        u = torch.clamp(u, -1, 1)
        tmp = u * 4 * std_t + mean_t

    
        new_st = st + 1 + torch.sum(components).item()  # Increment for next iteration
        return tmp, new_st, []




        
        
        

