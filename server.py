#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated for Flask framework implementation
"""
from flask import Flask, request, jsonify
import numpy as np
import models
import os
from src.server_utils import initialize_global_model, weights_to_list, list_to_weights, evaluate_global_model
from src.data_distribution import DataDistribution
from src.averaging_models import AveragingModels
from src.get_coefficient_weights import get_coefficients
import numpy as np 
import matplotlib.pyplot as plt
from src.plotting import plotter1, plotter2, plotter3


app = Flask(__name__)

# Initialize global model and data distribution
print("......Intializing global model")
global_model, global_weights = initialize_global_model()
print("......Loading data distribution")

data_distribution = DataDistribution(is_iid=False, is_weighted=True, inverse=True)
_, _, X_valid, y_valid = data_distribution.get_data()
valid_dist = data_distribution.get_valid_dist()
current_round = 0

accuracy_per_class_over_rounds = []
precision_per_class_over_rounds = []
recall_per_class_over_rounds = []
f1_per_class_over_rounds = []
round_numbers = []
client_accuracies = []

coeff_weights_list = []

with open("server_logs.txt", "w") as file:
    file.write("")

@app.route('/init', methods=['GET'])
def init_model():
    print('Sharing Initial Global Model with Random Weight Initialization')
    return jsonify(weights_to_list(global_weights))

@app.route('/update_weights', methods=['POST'])
def update_weights():
    global current_round, global_weights, accuracy_per_class_over_rounds, precision_per_class_over_rounds, recall_per_class_over_rounds, f1_per_class_over_rounds, round_numbers
    current_round += 1

    print(f"Training round {current_round}")
    print("Client training starts...")

    client_weights = [list_to_weights(w) for w in request.json['weights']]
    client_accuracies.append(request.json['client_accuracy'])
    
    if current_round == 1:
        coefficient_weights = data_distribution.get_initial_coefficient_weights()
    else:
        coefficient_weights = get_coefficients(coeff_weights_list, client_accuracies, current_round)

    print("\n\n", coefficient_weights)
    coeff_weights_list.append(coefficient_weights)

    with open("server_logs.txt", "a") as file:
        file.write(f"Round: {current_round}\n")
        file.write(str(coefficient_weights)+"\n")

    # avg_weights = AveragingModels.model_weighted_average(client_weights, coefficient_weights)
    avg_weights = AveragingModels.model_average_q0(client_weights)
    
    global_weights = avg_weights
    
    # Evaluate model
    loss, accuracy, accuracy_per_class, precision_per_class, recall_per_class, f1_per_class = evaluate_global_model(global_model, global_weights, X_valid, y_valid)
    
    with open("server_logs.txt", "a") as file:
        file.write(f"accuracy= {accuracy}, loss = {loss}\n")
        file.write(f"accuracy_per_class: {str(accuracy_per_class)}\n\n")

    accuracy_per_class_over_rounds.append(accuracy_per_class)
    precision_per_class_over_rounds.append(precision_per_class)
    recall_per_class_over_rounds.append(recall_per_class)
    f1_per_class_over_rounds.append(f1_per_class)
    round_numbers.append(current_round)

    response = {
        'round': current_round,
        'weights': weights_to_list(global_weights),
        'accuracy': accuracy,
        'loss': loss,
    }
    
    print(f"#######----- Accuracy for round {current_round} is {accuracy} and loss is {loss} ------########")
    # print(accuracy_per_class)
    # print(f"Precision, Recall and F1 Scores for round {current_round} is {precision_per_class}, {recall_per_class} and {f1_per_class}")
    
    return jsonify(response)


if __name__ == "__main__":
    print("...Initiating global model")
    print("---")
    app.run(port=5000)
    
    np.save("./npy results/accuracy_per_class_over_rounds.npy", np.array(accuracy_per_class_over_rounds))
    np.save("./npy results/precision_per_class_over_rounds.npy", np.array(precision_per_class_over_rounds))
    np.save("./npy results/recall_per_class_over_rounds.npy", np.array(recall_per_class_over_rounds))
    np.save("./npy results/f1_per_class_over_rounds.npy", np.array(f1_per_class_over_rounds))
    np.save("./npy results/round_numbers.npy", np.array(round_numbers))

    cur_dir_path = os.getcwd()
    # plotter1(cur_dir_path)
    # plotter2(cur_dir_path)
    # plotter3(cur_dir_path, valid_dist)