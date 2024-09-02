#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 02:44:26 2023

Updated for modular data distribution
"""
import load_data
from collections import Counter
import numpy as np

class DataDistribution:
    def __init__(self, beta=0.5, alpha_min=0.5, alpha_max=1.0):
        self.x_data, self.y_data, self.X_valid, self.y_valid, self.coefficient_weights, self.valid_dist = load_data.load_cicids_2017_new(is_iid = False, is_weighted = True, inverse = True)
        self.coeff_weights_list = []
        self.beta = beta
        self.n = len(self.coefficient_weights)
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

    def get_data(self):
        return self.x_data, self.y_data, self.X_valid, self.y_valid
    
    def get_valid_dist(self):
        return self.valid_dist

    def find_q_updated(self, client_accuracies, current_round):
        if len(self.coefficient_weights) != 0:
            if current_round == 1:
                self.coeff_weights_list.append(self.coefficient_weights)
                return self.coefficient_weights
            else:
                return self.new_coeff_weights(client_accuracies, current_round)
        else:
            self.find_q()


    def arw(self, current_accuracy, previous_accuracy, current_coeff_weight):
        delta_accuracy = current_accuracy - previous_accuracy
        
        alpha = 1 - self.beta * delta_accuracy
        
        alpha = np.clip(alpha, self.alpha_min, self.alpha_max)
        
        # Adjust the reparameterization weight
        new_coeff_weight = current_coeff_weight * alpha
        
        return new_coeff_weight


    def new_coeff_weights(self, client_accuracies, current_round):
        new_weights = []
        for i in range(self.n):
            new_weights.append(self.arw(client_accuracies[current_round-1][i], client_accuracies[current_round-2][i], self.coeff_weights_list[current_round-2][i]))
        self.coeff_weights_list.append(new_weights)
        return new_weights

    def find_q(self):
        p_all_node_list = []
        # Here malicious nodes are 3,5,7
        malicious_nodes = [3, 5, 7]
        for index in range(len(self.y_data)):
            counter_num = Counter(self.y_data[index])
            total_sample = len(self.y_data[index])
            try:
                benign_sample = counter_num[0]
            except:
                p_all_node_list.append(0)
                continue
            malicious_sample = total_sample - benign_sample
            p_node = malicious_sample / benign_sample
            p_all_node_list.append(p_node)

        q_all_list_node = self.softmax(p_all_node_list)
        return q_all_list_node

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    
    
    
    