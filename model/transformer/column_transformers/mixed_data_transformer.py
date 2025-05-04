import numpy as np
import pandas as pd
import torch
from sklearn.mixture import BayesianGaussianMixture
from .column_transformer import Column_transformer

class Mixed_data_transformer(Column_transformer):

    def __init__(self,column,modals,n_clusters=10,eps=0.005):
        super().__init__()
        self.modals = modals
        self.n_clusters = n_clusters
        self.eps = eps
        self.min = column.min()
        self.max = column.max()
        self.nan_replacement = -9999999

        self.fit(column.to_numpy())

    def fit(self, data_col):

        gm1 = BayesianGaussianMixture(
            n_components = self.n_clusters, 
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001, max_iter=100,
            n_init=1,random_state=42)
        gm2 = BayesianGaussianMixture(
            n_components = self.n_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001, max_iter=100,
            n_init=1,random_state=42)

        if np.nan in self.modals: self.modals[self.modals.index(np.nan)] = self.nan_replacement
        data_col = np.where(np.isnan(data_col), self.nan_replacement, data_col)
        gm1.fit(data_col.reshape([-1, 1]))

        self.filter_arr = ~np.isin(data_col, self.modals) # Find the indices of elements in data[:, id_] that are not in modal (aka that are not in the normal continuous data)
        data_continuous = self.get_continuous_data(data_col, self.filter_arr) # The observations where we have continuous data, excluding the modal/categorical data
        gm2.fit(data_continuous)

        component_assignments = gm2.predict(data_continuous) # Assigns each observation to a component

        mode_freq = pd.Series(component_assignments).value_counts().keys() # Calculate the frequency of each component assignment

        # self.filter_arr.append(filter_arr)
        self.model = gm2
        self.model0 = gm1 #TODO: remove thsi is tamp
       
        old_comp = gm2.weights_ > self.eps # To prevent the model from overfitting? removes components that are less than eps
          
        # The final components are the ones that are frequent and have a weigh greater than the eps cutoff
        self.components = [(i in mode_freq) & old_comp[i] for i in range(self.n_clusters)]


        self.output_info = [(1, 'tanh',"no_g"), (np.sum(self.components) + len(self.modals), 'softmax', None)]
        self.output_dim = 1 + np.sum(self.components) + len(self.modals)

    def transform(self, data_col):
        data_size = len(data_col)
        

        data_col = np.where(np.isnan(data_col), self.nan_replacement, data_col)
        filter_arr = self.filter_arr
        data_filter_col = self.get_continuous_data(data_col, self.filter_arr)
        
        

        #####################################

        means_0 = self.model0.means_.reshape([-1])
        stds_0 = np.sqrt(self.model0.covariances_).reshape([-1])

        zero_std_list = []
        means_needed = []
        stds_needed = []

        for mode in self.modals:
            if mode!=-9999999:
                dist = []
                for idx,val in enumerate(list(means_0.flatten())):
                    dist.append(abs(mode-val))
                index_min = np.argmin(np.array(dist))
                zero_std_list.append(index_min)
            else: continue

        for idx in zero_std_list:
            means_needed.append(means_0[idx])
            stds_needed.append(stds_0[idx])
        
        
        mode_vals = []

        for i,j,k in zip(self.modals,means_needed,stds_needed):
            this_val  = np.abs(i - j) / (4*k)
            mode_vals.append(this_val)
        
        if -9999999 in self.modals:
            mode_vals.append(0)

        #############################

        means = self.model.means_.reshape((1, self.n_clusters))
        stds = np.sqrt(self.model.covariances_).reshape((1, self.n_clusters))
        features = np.empty(shape=(len(data_filter_col),self.n_clusters))
        
        features = (data_filter_col - means) / (4 * stds)

        probs = self.model.predict_proba(data_filter_col.reshape([-1, 1]))

        n_opts = sum(self.components) 
        features = features[:, self.components]
        probs = probs[:, self.components]
        
        opt_sel = np.zeros(len(data_filter_col), dtype='int')
        for i in range(len(data_filter_col)):
            pp = probs[i] + 1e-6
            pp = pp / sum(pp)
            opt_sel[i] = np.random.choice(np.arange(n_opts), p=pp)
        idx = np.arange((len(features)))
        features = features[idx, opt_sel].reshape([-1, 1])
        features = np.clip(features, -.99, .99)
        probs_onehot = np.zeros_like(probs)
        probs_onehot[np.arange(len(probs)), opt_sel] = 1
        extra_bits = np.zeros([len(data_filter_col), len(self.modals)])
        temp_probs_onehot = np.concatenate([extra_bits,probs_onehot], axis = 1)
        final = np.zeros([len(data_col), 1 + probs_onehot.shape[1] + len(self.modals)])
        features_curser = 0
        for idx, val in enumerate(data_col): #TODO: This should be as data_col instead, since now we are using the original data, this might fix the pr
            if val in self.modals:
                category_ = list(map(self.modals.index, [val]))[0]
                final[idx, 0] = mode_vals[category_]
                final[idx, (category_+1)] = 1
            
            else:
                final[idx, 0] = features[features_curser]
                final[idx, (1+len(self.modals)):] = temp_probs_onehot[features_curser][len(self.modals):]
                features_curser = features_curser + 1
        
        just_onehot = final[:,1:]
        re_ordered_jhot= np.zeros_like(just_onehot)
        n = just_onehot.shape[1]
        col_sums = just_onehot.sum(axis=0)
        largest_indices = np.argsort(-1*col_sums)[:n]
        self.ordering = largest_indices
        for id,val in enumerate(largest_indices):
            re_ordered_jhot[:,id] = just_onehot[:,val]
        final_features = final[:,0].reshape([-1, 1])
        return np.concatenate([final_features, re_ordered_jhot], axis=1)
        

    def inverse_transform(self, data,st):
        u = data[:, st]
        full_v = data[:,(st+1):(st+1)+len(self.modals)+np.sum(self.components)]
        
        full_v_re_ordered = np.zeros_like(full_v)

        for id,val in enumerate(self.ordering):
            full_v_re_ordered[:,val] = full_v[:,id]
            
        full_v = full_v_re_ordered

        
        mixed_v = full_v[:,:len(self.modals)]
        v = full_v[:,-np.sum(self.components):]

        u = np.clip(u, -1, 1)
        v_t = np.ones((data.shape[0], self.n_clusters)) * -100
        v_t[:, self.components] = v
        v = np.concatenate([mixed_v,v_t], axis=1)

        

        means = self.model.means_.reshape([-1]) 
        stds = np.sqrt(self.model.covariances_).reshape([-1]) 
        p_argmax = np.argmax(v, axis=1)

        result = np.zeros_like(u)

        for idx in range(len(data)):
            if p_argmax[idx] < len(self.modals):
                argmax_value = p_argmax[idx]
                result[idx] = float(list(map(self.modals.__getitem__, [argmax_value]))[0])
            else:
                std_t = stds[(p_argmax[idx]-len(self.modals))]
                mean_t = means[(p_argmax[idx]-len(self.modals))]
                result[idx] = u[idx] * 4 * std_t + mean_t
        
        result[result == self.nan_replacement] = np.nan
        invalid_ids = np.where((result < self.min) | (result > self.max))[0].tolist()
        
        new_st = 1 + np.sum(self.components) + len(self.modals)
        return result, new_st, invalid_ids


    def get_continuous_data(self,data_col,filter_arr):
        data_col = np.where(np.isnan(data_col), self.nan_replacement, data_col)
        return data_col[filter_arr].reshape([-1, 1])


    def inverse_transform_static(self, data, transformer, st, device, n_clusters=10):
        components = transformer.get_component()
        order = transformer.get_ordering()
        model = transformer.get_model()
        modals = transformer.modals

        components = torch.tensor(components, dtype=torch.bool).to(device)
        u = data[:, st]
        full_v = data[:, (st + 1):(st + 1) + len(modals) + torch.sum(components).item()]

        # Ensure order is a tensor
        order = torch.tensor(order, device=device)
        full_v = full_v[:, torch.argsort(order)]

        mixed_v = full_v[:, :len(modals)]
        v = full_v[:, -torch.sum(components).item():]

        u = torch.clamp(u, -1, 1)
        v_t = torch.ones((data.shape[0], n_clusters), device=device) * -float('inf')
        v_t[:, components] = v
        v = torch.cat([mixed_v, v_t], dim=1)

        means = torch.tensor(model[1].means_.reshape([-1]), device=device)
        stds = torch.sqrt(torch.tensor(model[1].covariances_.reshape([-1]), device=device))
        p_argmax = torch.argmax(v, dim=1)

        result = torch.zeros_like(u)
        for idx in range(len(data)):
            if p_argmax[idx] < len(modals):
                argmax_value = p_argmax[idx]
                result[idx] = float(list(map(modals.__getitem__, [argmax_value]))[0])
            else:
                std_t = stds[(p_argmax[idx] - len(modals))]
                mean_t = means[(p_argmax[idx] - len(modals))]
                result[idx] = u[idx] * 4 * std_t + mean_t

        invalid_ids = torch.where((result < transformer.min) | (result > transformer.max))[0].tolist()
        new_st = 1 + torch.sum(components).item() + len(modals)
        return result, new_st, invalid_ids

        
        

        