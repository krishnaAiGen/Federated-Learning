#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated for Flask framework implementation
"""
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os
from src.server_utils import initialize_global_model, weights_to_list, list_to_weights, evaluate_global_model
from src.data_distribution import DataDistribution
from src.averaging_models import AveragingModels
from src.get_coefficient_weights import get_coefficients
import numpy as np 
import matplotlib.pyplot as plt
from src.plotting import *
from collections import Counter
from src.calculate_divergence import *
from models import get_cicids_model

import tensorflow as tf
from tensorflow import keras


reparam_methods = ["fedavg", "static", "adaptive"]
option = 2
reparam_method = reparam_methods[option-1]

alpha = 1

app = Flask(__name__)

# Initialize global model and data distribution
print("......Intializing global model")
global_model, global_weights = initialize_global_model(initialise_new=False, save=False)
print("......Loading data distribution")

last_k = 3

if reparam_method == "adaptive":
    data_distribution = DataDistribution(is_iid=False, is_weighted=True, inverse=True, k = last_k)
elif reparam_method == "static":
    data_distribution = DataDistribution(is_iid=False, is_weighted=True, inverse=True, k = last_k)
elif reparam_method == "fedavg":
    data_distribution = DataDistribution(is_iid=False, is_weighted=True, inverse=False, k = last_k)

_, _, smallest_k_ids, X_valid, y_valid = data_distribution.get_data()
valid_dist = data_distribution.get_valid_dist()
current_round = 0

print(smallest_k_ids)

indices = np.where(np.isin(y_valid, smallest_k_ids))[0]

X_valid_last_k = np.array([X_valid[i] for i in indices])
y_valid_last_k = y_valid[indices]

x_valid_tf = tf.convert_to_tensor(X_valid, dtype=tf.float32)
y_valid_tf = tf.convert_to_tensor(y_valid, dtype=tf.int32)

x_valid_tf_last_k = tf.convert_to_tensor(X_valid_last_k, dtype=tf.float32)
y_valid_tf_last_k = tf.convert_to_tensor(y_valid_last_k, dtype=tf.int32)

accuracy_per_class_over_rounds = []
precision_per_class_over_rounds = []
recall_per_class_over_rounds = []
f1_per_class_over_rounds = []
round_numbers = []
client_accuracies = []
confusion_matrix_list = []

divergence_list = []


coeff_weights_list = []

centrailsed_weights = np.load('centrailsed_weights.npz', allow_pickle=True)

def filter_dataset_by_label(x_valid, y_valid, desired_labels):

    # Ensure desired_labels is a list for consistency
    if isinstance(desired_labels, int):
        desired_labels = [desired_labels]

    # Find indices where y_valid matches the desired label(s)
    indices = np.isin(y_valid, desired_labels)

    # Filter x_valid and y_valid based on the indices
    new_x_valid = x_valid[indices]
    new_y_valid = y_valid[indices]

    return new_x_valid, new_y_valid


client_valid_data = []

for id in smallest_k_ids:
    filtered_x, filtered_y = filter_dataset_by_label(X_valid, y_valid, id)
    client_valid_data.append([filtered_x, filtered_y])


def get_normalized_weights(w):
    return tf.nn.softmax(w)


def aggregate_client_weights(client_weights_list, w):
    normalized_w = tf.nn.softmax(w)
    n_clients = len(client_weights_list)

    # Get the shapes of the client model weights
    model_weights_shapes = [var.shape for var in client_weights_list[0]]

    # Initialize aggregated_weights as tensors of zeros with appropriate shapes
    aggregated_weights = [
        tf.zeros(shape, dtype=tf.float32) for shape in model_weights_shapes
    ]

    for i in range(n_clients):
        weight = normalized_w[i]  # Scalar tensor
        client_vars = client_weights_list[i]  # List of tensors (weights)

        for j in range(len(aggregated_weights)):
            aggregated_weights[j] += weight * client_vars[j]

    return aggregated_weights  # List of tensors

# Ensure forward_pass function is included
def forward_pass(x, aggregated_weights):
    # Unpack the aggregated weights
    w1, b1, w2, b2, w3, b3, w4, b4 = aggregated_weights

    # Layer 1
    x = tf.matmul(x, w1) + b1
    x = tf.nn.tanh(x)

    # Layer 2
    x = tf.matmul(x, w2) + b2
    x = tf.nn.tanh(x)

    # Layer 3
    x = tf.matmul(x, w3) + b3
    x = tf.nn.tanh(x)

    # Output Layer
    x = tf.matmul(x, w4) + b4
    x = tf.nn.softmax(x)

    return x



with open("server_logs.txt", "w") as file:
    file.write("")



@app.route('/init', methods=['GET'])
def init_model():
    print('Sharing Initial Global Model with Random Weight Initialization')
    return jsonify(weights_to_list(global_weights))


@tf.function
@app.route('/update_weights', methods=['POST'])
def update_weights():
    global current_round, global_weights, accuracy_per_class_over_rounds, precision_per_class_over_rounds, recall_per_class_over_rounds, f1_per_class_over_rounds, round_numbers, wgt
    current_round += 1

    print(f"Training round {current_round}")
    print("Client training starts...")

    client_weights = [list_to_weights(w) for w in request.json['weights']]
    client_accuracies.append(request.json['client_accuracy'])

    client_weights_list = []
    for client_weights_single in client_weights:
        client_weights_tensors = [
            tf.convert_to_tensor(w, dtype=tf.float32) for w in client_weights_single
        ]
        client_weights_list.append(client_weights_tensors)


    if reparam_method == "adaptive":    
        if current_round == 1:
            coefficient_weights = data_distribution.get_initial_coefficient_weights()
        else:
            coefficient_weights = get_coefficients(coeff_weights_list, client_accuracies, current_round)
    else:
        if current_round == 1:
            coefficient_weights = data_distribution.get_initial_coefficient_weights()
            print("Iniital Coefficient Weights: \n")
            print(coefficient_weights)
        else:
            coefficient_weights = wgt

    initial_weights = np.array(coefficient_weights, dtype=np.float32)
    wgt = tf.Variable(initial_weights, dtype=tf.float32)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    num_epochs = 5

    for epoch in range(num_epochs):
        with tf.GradientTape() as tape:
            tape.watch(wgt)
            # Aggregate client weights
            aggregated_weights = aggregate_client_weights(client_weights_list, wgt)
            # Forward pass
            predictions_all = forward_pass(x_valid_tf, aggregated_weights)
            # Compute loss
            loss_all = loss_fn(y_valid_tf, predictions_all)

            predictions_last_k = forward_pass(x_valid_tf_last_k, aggregated_weights)
            # Compute loss
            loss_last_k = loss_fn(y_valid_tf_last_k, predictions_last_k)


            loss = alpha*loss_all + (1-alpha)*loss_last_k


        # Compute gradients with respect to w
        grads = tape.gradient(loss, wgt)

        if grads is not None:
            optimizer.apply_gradients([(grads, wgt)])
        else:
            print("Gradients are None.")

        # Normalize weights
        # wgt.assign(tf.nn.softmax(wgt))

        # wgt_sum = tf.reduce_sum(wgt)

        # # Normalize each element in wgt by dividing by the sum
        # wgt.assign(wgt / wgt_sum)

        # wgt_sum_of_squares = tf.reduce_sum(tf.square(wgt))

        # wgt.assign(wgt / wgt_sum_of_squares)

        # wgt_abs_sum = tf.reduce_sum(tf.abs(wgt))

        # wgt.assign(wgt / wgt_abs_sum)

        # wgt_mean = tf.reduce_mean(wgt)
        # wgt_centered  = wgt - wgt_mean

        # wgt_stddev = tf.math.reduce_std(wgt_centered)

        # wgt.assign(wgt_centered / wgt_stddev)

        print(f"Epoch {epoch+1}, Loss: {loss.numpy()}, Weights: {wgt.numpy()}")

    # Compute the final aggregated weights
    aggregated_weights = aggregate_client_weights(client_weights_list, wgt)
    global_weights = [w.numpy() for w in aggregated_weights]  # Convert tensors to NumPy arrays

    # Update coefficient weights list

    coeff_weights_list.append(wgt.numpy().tolist())

    global_model = get_cicids_model()
    global_model.set_weights(global_weights)

    # print(coefficient_weights)

    with open("server_logs.txt", "a") as file:
        file.write(f"Round: {current_round}\n")
        file.write(str(coefficient_weights)+"\n")

    # avg_weights = AveragingModels.model_weighted_average(client_weights, coefficient_weights)
    
    # global_weights = avg_weights

    with open("underrepresented_client_logs.txt", "a") as file:
            file.write(f"Round: {current_round}\n")

    for h in range(len(smallest_k_ids)):
        client_id = 13-h
        filtered_x, filtered_y = client_valid_data[h]
        loss, accuracy, accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, confusion_matrix = evaluate_global_model(global_model, client_weights[client_id], filtered_x, filtered_y)
        with open("underrepresented_client_logs.txt", "a") as file:
            file.write(f"Client: {client_id}\n")
            file.write(f"loss: {loss}\n")
            file.write(f"accuracy: {accuracy}\n")

    with open("underrepresented_client_logs.txt", "a") as file:
        file.write("\n")
    
    divergence = compute_euclidean_distance(centrailsed_weights, global_weights)
    divergence_list.append(divergence)

    # Evaluate model
    loss, accuracy, accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, confusion_matrix = evaluate_global_model(global_model, global_weights, X_valid, y_valid)
    confusion_matrix_list.append(confusion_matrix)

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

    experiment_name = f"{reparam_method} final {alpha}"

    output_dir = f"{experiment_name}/npy results"
    os.makedirs(output_dir, exist_ok=True)

    # Save to .npy file
    np.save(f"./{experiment_name}/npy results/confusion_matrices.npy", np.array(confusion_matrix_list))
    np.save(f"./{experiment_name}/npy results/accuracy_per_class_over_rounds.npy", np.array(accuracy_per_class_over_rounds))
    np.save(f"./{experiment_name}/npy results/precision_per_class_over_rounds.npy", np.array(precision_per_class_over_rounds))
    np.save(f"./{experiment_name}/npy results/recall_per_class_over_rounds.npy", np.array(recall_per_class_over_rounds))
    np.save(f"./{experiment_name}/npy results/f1_per_class_over_rounds.npy", np.array(f1_per_class_over_rounds))
    np.save(f"./{experiment_name}/npy results/round_numbers.npy", np.array(round_numbers))

    cur_dir_path = os.getcwd()
    plotter1(cur_dir_path, experiment_name)
    plotter2(cur_dir_path, experiment_name)
    plotter3(cur_dir_path, valid_dist, experiment_name)
    plotter4(cur_dir_path, smallest_k_ids, experiment_name)
    plotter5(range(1,current_round+1), divergence_list, experiment_name)