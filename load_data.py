#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 03:25:41 2023

@author: krishna
"""


"""
This function is to prevent loading the data from local computer multiple times.
"""
# def retreive_data(): 
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
   
def retreive_data():
    X_train = pd.read_csv('./Data/cicids-2017/cicids_x_train.csv')
    X_valid = pd.read_csv('./Data/cicids-2017/cicids_x_test.csv')
    y_train = np.load('./Data/cicids-2017/cicids-y-train.npy')
    y_valid = np.load('./Data/cicids-2017/cicids-y-test.npy')
    
    return X_train, X_valid, y_train, y_valid


def sample_data(X_train, y_train, benign_label=0, sample_size=10000):
    # Convert X_train and y_train to a single DataFrame for easier manipulation
    df = pd.DataFrame(X_train)
    df['Label'] = y_train

    sampled_dfs = []
    benign_samples_needed = 0

    # Get the list of unique labels
    labels = df['Label'].unique()

    for label in labels:
        if label != benign_label:
            label_df = df[df['Label'] == label]
            num_samples = min(len(label_df), sample_size)
            sampled_df = label_df.sample(n=num_samples, random_state=42)
            sampled_dfs.append(sampled_df)
            benign_samples_needed += num_samples


    benign_df = df[df['Label'] == benign_label]
    if len(benign_df) >= benign_samples_needed:
        benign_sampled_df = benign_df.sample(n=benign_samples_needed, random_state=42)
    else:
        benign_sampled_df = benign_df  # If not enough samples, take all

    sampled_dfs.append(benign_sampled_df)

    # Combine all the sampled dataframes
    final_df = pd.concat(sampled_dfs, ignore_index=True)

    # Separate the features and labels again
    X_sampled = final_df.drop(columns=['Label']).to_numpy()
    y_sampled = final_df['Label'].to_numpy()

    return X_sampled, y_sampled


def stratified_split(X_combined, y_combined, test_size=0.1):
    X_train_list = []
    y_train_list = []
    X_valid_list = []
    y_valid_list = []

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(X_combined)
    df['Label'] = y_combined

    # Get the list of unique labels
    labels = df['Label'].unique()

    for label in labels:
        label_df = df[df['Label'] == label]
        
        # Perform a stratified split for this label's data
        X_train_label, X_valid_label, y_train_label, y_valid_label = train_test_split(
            label_df.drop(columns=['Label']), label_df['Label'], 
            test_size=test_size, random_state=42, stratify=label_df['Label'])
        
        # Append the splits to the corresponding lists
        X_train_list.append(X_train_label)
        y_train_list.append(y_train_label)
        X_valid_list.append(X_valid_label)
        y_valid_list.append(y_valid_label)

    # Combine all the lists to form the final training and validation sets
    X_train = pd.concat(X_train_list).to_numpy()
    y_train = pd.concat(y_train_list).to_numpy()
    X_valid = pd.concat(X_valid_list).to_numpy()
    y_valid = pd.concat(y_valid_list).to_numpy()

    return X_train, X_valid, y_train, y_valid


import numpy as np

def percentage_imbalance(X_combined, y_combined, p=0.5, num_clients=14):
    # Number of imbalance clients
    imbalance_clients = round(p * num_clients)

    # Count unique labels and their occurrences in y_combined
    unique_labels, label_counts = np.unique(y_combined, return_counts=True)

    # Get the least frequent labels
    least_frequent_indices = np.argsort(label_counts)[:imbalance_clients]
    least_frequent_labels = unique_labels[least_frequent_indices]

    # Create imbalanced dataset
    X_imbalanced = []
    y_imbalanced = []

    for label in least_frequent_labels:
        label_indices = np.where(y_combined == label)[0]
        n_samples = min(35, len(label_indices))
        
        # Take only 'n_samples' from the least frequent labels
        selected_indices = label_indices[:n_samples]
        X_imbalanced.append(X_combined[selected_indices])
        y_imbalanced.append(y_combined[selected_indices])

    # Append unaffected samples (those from labels not part of least frequent clients)
    unaffected_indices = np.isin(y_combined, least_frequent_labels, invert=True)
    X_imbalanced.append(X_combined[unaffected_indices])
    y_imbalanced.append(y_combined[unaffected_indices])
    
    # Concatenate the lists back into numpy arrays
    X_imbalanced = np.vstack(X_imbalanced)
    y_imbalanced = np.hstack(y_imbalanced)
    
    return X_imbalanced, y_imbalanced


"""
This function loads data where normal attack and malicious attack
are mixed together and distributed in all the node
"""

def load_cicids_2017(k):
    # Load the data
    X_train, X_valid, y_train, y_valid = retreive_data()

    # Combine the training and validation datasets
    X_combined = pd.concat([X_train, X_valid])
    y_combined = np.concatenate([y_train, y_valid])

    X_combined, y_combined = sample_data(X_combined, y_combined, benign_label=0, sample_size=10000)

    X_combined, y_combined = percentage_imbalance(X_combined, y_combined, p = 0, num_clients = 14)



    X_train, X_valid, y_train, y_valid = stratified_split(X_combined, y_combined, test_size=0.1)

    dist_y = Counter(y_train)

    smallest_k_ids = sorted(dist_y, key=dist_y.get)[:k]

    return X_train, X_valid, y_train, y_valid, smallest_k_ids


"""
This function loads data where malicious attack are distributed in only certain nodes
where some amount of benign traffic is also present. The ratio of malicious to benign
traffic in these nodes is 9:1. and other normal traffic are distributed in
non-malicious nodes. If K is total nodes then K_b are benign nodes and K_m are 
malicious nodes.
"""

def load_cicids_2017_q():
    X_train, X_valid, y_train, y_valid = retreive_data()
    X_train['Label'] = y_train
    X_train_full = X_train
    X_train = X_train.drop(['Label'], axis = 1)
    
    #In this dataset 0 represents benign traffic and other value represents other.
    X_malicious = X_train_full[X_train_full['Label']!=0]
    X_benign = X_train_full[X_train_full['Label']==0]
    
    attack_dictionary = {}
    
    #selecting only subset of data as there are too much data in valiation set.
    X_valid = X_valid[1000:5000]
    y_valid = y_valid[1000:5000]                
        
    """
    Nodes 4,6,8 are malicious
    """
    
    malicious_nodes = [4,6,8]
    X_train_list = []
    y_train_list = []
    for index in range(1,11):
        if index in malicious_nodes:
            dataset_malicious = X_malicious[index*10000:(index+1)*10000]
            dataset_benign = X_benign[index*2000:(index+1)*2000]
            dataset_joined = pd.concat([dataset_benign, dataset_malicious])
            dataset_joined = dataset_joined.sample(frac=1)
            y_train_list.append(dataset_joined['Label'])
            X_train_list.append(dataset_joined.drop(['Label'], axis = 1))
        
        else:
            dataset_joined = X_benign[index*2000:(index+1)*2000]
            dataset_joined = dataset_joined.sample(frac=1)
            y_train_list.append(dataset_joined['Label'])
            X_train_list.append(dataset_joined.drop(['Label'], axis = 1))
            

    
    return X_train_list, y_train_list, X_valid, y_valid





# def load_nsl_kdd():
    