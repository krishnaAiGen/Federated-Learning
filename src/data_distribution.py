#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 02:44:26 2023

Updated for modular data distribution
"""
import load_data
from collections import Counter
import numpy as np
import pandas as pd

class DataDistribution:
    def __init__(self, is_iid=False, is_weighted=True, inverse=True, k=0):
        self.is_iid = is_iid
        self.is_weighted = is_weighted
        self.inverse = inverse

        self.x_data, self.y_data, self.X_valid, self.y_valid, self.coefficient_weights, self.valid_dist, self.smallest_k_ids = self.fetch_data_and_weights(k)

    def fetch_data_and_weights(self, k):
        if self.is_iid:
            return self.load_cicids_2017_iid(k)
        else:
            return self.load_cicids_2017_non_iid(k)
        
    def get_data(self):
        return self.x_data, self.y_data, self.smallest_k_ids, self.X_valid, self.y_valid
    
    def get_initial_coefficient_weights(self):
        return self.coefficient_weights
    
    def get_valid_dist(self):
        return self.valid_dist

    def load_cicids_2017_non_iid(self, k):
        X_combined, X_valid, y_combined, y_valid, smallest_k_ids = load_data.load_cicids_2017(k)
        
        X_combined_df = pd.DataFrame(X_combined)
        X_combined_df['Label'] = y_combined
        
        X_malicious = X_combined_df[X_combined_df['Label'] != 0]
        X_benign = X_combined_df[X_combined_df['Label'] == 0]
        
        valid_dist = Counter(y_valid)

        X_train_list = []
        y_train_list = []
        client_sample_counts = []

        labels = X_malicious['Label'].unique()
        
        for index, label in enumerate(labels):
            dataset_malicious = X_malicious[X_malicious['Label'] == label]
            
            num_malicious_samples = len(dataset_malicious)
            dataset_benign = X_benign.sample(n=num_malicious_samples, random_state=42)
            
            dataset_joined = pd.concat([dataset_malicious, dataset_benign])
            dataset_joined = dataset_joined.sample(frac=1, random_state=42)

            X_train = dataset_joined.drop(columns=['Label']).to_numpy()
            y_train = dataset_joined['Label'].to_numpy()
            
            X_train_list.append(X_train)
            y_train_list.append(y_train)
            
            if self.is_weighted:
                client_sample_counts.append(len(y_train))
            else:
                client_sample_counts.append(1)

        if self.is_weighted and self.inverse:
            total_samples = sum(client_sample_counts)
            
            coefficient_weights = [total_samples / count for count in client_sample_counts]
            # coefficient_weights = [alpha*(count / total_samples) + beta*(total_samples / count) for count in client_sample_counts]

            total_coeff = sum(coefficient_weights)

            coefficient_weights = [weight / total_coeff for weight in coefficient_weights]
        else:
            if self.is_weighted:
                total_samples = sum(client_sample_counts)
                
                coefficient_weights = [count / total_samples for count in client_sample_counts]
            else:
                coefficient_weights = [1/len(client_sample_counts) for cnt in client_sample_counts]

        return X_train_list, y_train_list, X_valid, y_valid, coefficient_weights, valid_dist, smallest_k_ids

    def load_cicids_2017_iid(self, k):
        X_combined, X_valid, y_combined, y_valid, smallest_k_ids = load_data.load_cicids_2017(k)
        
        X_combined_df = pd.DataFrame(X_combined)
        X_combined_df['Label'] = y_combined

        valid_dist = Counter(y_valid)
        
        X_combined_df = X_combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        num_clients = 14
        
        client_data = np.array_split(X_combined_df, num_clients)
        
        X_train_list = []
        y_train_list = []
        client_sample_counts = []
        
        for client_df in client_data:
            X_train = client_df.drop(columns=['Label']).to_numpy()
            y_train = client_df['Label'].to_numpy()
            
            X_train_list.append(X_train)
            y_train_list.append(y_train)
        
            if self.is_weighted:
                client_sample_counts.append(len(y_train))
            else:
                client_sample_counts.append(1)

        if self.is_weighted and self.inverse:
            total_samples = sum(client_sample_counts)
            
            coefficient_weights = [total_samples / count for count in client_sample_counts]

            total_coeff = sum(coefficient_weights)

            coefficient_weights = [weight / total_coeff for weight in coefficient_weights]

        else:
            if self.is_weighted:
                total_samples = sum(client_sample_counts)
                
                coefficient_weights = [count / total_samples for count in client_sample_counts]
            else:
                coefficient_weights = [1/len(client_sample_counts) for cnt in client_sample_counts]

        return X_train_list, y_train_list, X_valid, y_valid, coefficient_weights, valid_dist, smallest_k_ids