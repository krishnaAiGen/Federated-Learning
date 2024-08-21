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
        # coefficient_weights = [0.15650540813132544, 0.15650540813132544, 0.15650540813132544, 0.15650540813132544, 0.09280770702187598, 0.08426946753382256, 0.08181754947309845, 0.02300629499530484, 0.05037735192849442, 0.03048377560602372, 0.010190241018328522, 0.0005564636733558238, 0.00031301081626265085, 0.00015650540813132543]
        # coefficient_weights = [0.0005491161914100519, 0.0005491161914100519, 0.0005491161914100519, 0.0005491161914100519, 0.0009259969501012678, 0.0010198195878436786, 0.0010503816626334678, 0.003735484295306475, 0.0017059184406939822, 0.0028191932245809847, 0.008433525124045164, 0.1544389288340771, 0.27455809570502593, 0.5491161914100519]
        
        coefficient_weights = q_all_list_node

        print('Initiating averaging with coefficient weights')
        average_weight_list = []

        for index1 in range(len(client_weights[0])):
            layer_weights = [client_weights[index2][index1] for index2 in range(len(client_weights))]
            # Apply coefficient weights for weighted averaging
            layer_weights_weighted = [layer_weights[index3] * coefficient_weights[index3] for index3 in range(len(coefficient_weights))]
            average_weight = np.sum(np.array(layer_weights_weighted), axis=0)
            average_weight_list.append(average_weight)
        
        return average_weight_list

