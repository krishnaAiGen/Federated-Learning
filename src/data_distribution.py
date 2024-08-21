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
    def __init__(self):
        self.x_data, self.y_data, self.X_valid, self.y_valid, self.coefficient_weights, self.valid_dist = load_data.load_cicids_2017_new(is_iid = False, is_weighted = True, inverse = True)

    def get_data(self):
        return self.x_data, self.y_data, self.X_valid, self.y_valid
    
    def get_valid_dist(self):
        return self.valid_dist

    def find_q_updated(self):
        if len(self.coefficient_weights) != 0:
            return self.coefficient_weights
        else:
            self.find_q()

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