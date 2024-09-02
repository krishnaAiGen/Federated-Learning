#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 02:44:26 2023

Updated for modular averaging models
"""

import numpy as np

class AveragingModels:
    @staticmethod
    def model_average_q0(client_weights):
        average_weight_list = []
        for index1 in range(len(client_weights[0])):
            layer_weights = [client_weights[index2][index1] for index2 in range(len(client_weights))]
            average_weight = np.mean(np.array(layer_weights), axis=0)
            average_weight_list.append(average_weight)
        return average_weight_list
    

    @staticmethod
    def model_average_q1(client_weights, q_all_list_node):
        print('Initiating averagiing with q1')
        average_weight_list=[]
        
        for index1 in range(len(client_weights[0])):
            layer_weights=[]
            for index2 in range(len(client_weights)):
                weights=client_weights[index2][index1]
                layer_weights.append(weights)
                
                
            layer_weights_parameterized = []
            for index3 in range(len(q_all_list_node)):
                layer_weights_parameterized.append(layer_weights[index3]*q_all_list_node[index3])
            
            average_weight=np.sum(np.array([x for x in layer_weights_parameterized]), axis=0)
            average_weight_list.append(average_weight)
            
                    
        return average_weight_list

    @staticmethod
    def model_weighted_average(client_weights, coefficient_weights):
        print('Initiating averaging with coefficient weights')
        average_weight_list = []

        for index1 in range(len(client_weights[0])):
            layer_weights = [client_weights[index2][index1] for index2 in range(len(client_weights))]
            # Apply coefficient weights for weighted averaging
            layer_weights_weighted = [layer_weights[index3] * coefficient_weights[index3] for index3 in range(len(coefficient_weights))]
            average_weight = np.sum(np.array(layer_weights_weighted), axis=0)
            average_weight_list.append(average_weight)
        
        return average_weight_list