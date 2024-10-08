#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated for Flask framework implementation
"""
import os
import time
import requests
import numpy as np
import models
from src.client_training import Client, train_server_with_parallelization, train_server_without_parallelization
from src.data_distribution import DataDistribution
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

if __name__ == "__main__":
    epochs = 2
    lr = 0.01
    batch_size = 64
    rounds = 100
    use_parallel = True

    start_time = time.time()

    print("...Loading data distribution at client side")
    data_distribution = DataDistribution()
    x_data, y_data, _, _ = data_distribution.get_data()
    num_clients = len(y_data)
    clients = [Client(x_data[i], y_data[i], epoch_number=epochs, learning_rate=lr, batch=batch_size) for i in range(num_clients)]

    # Get initial global weights from server
    response = requests.get('http://127.0.0.1:5000/init')
    global_weights = [np.array(w) for w in response.json()]

    if use_parallel:
        print("Training with parallelization")
        training_accuracy, loss_list = train_server_with_parallelization(rounds, clients, global_weights, 'http://127.0.0.1:5000')
        np.save('accuracy_cicids_fedq_parallel', np.array(training_accuracy))
        np.save('loss_cicids_fedq_parallel', np.array(loss_list))
    else:
        print("Training without parallelization")
        training_accuracy, loss_list = train_server_without_parallelization(rounds, clients, global_weights, 'http://127.0.0.1:5000')
        np.save('accuracy_cicids_fedq', np.array(training_accuracy))
        np.save('loss_cicids_fedq', np.array(loss_list))

    end_time = time.time()
    print('TOTAL TIME ELAPSED = ', end_time - start_time)

    # Plotting the loss
    from src.plotting import plot_loss
    loss_fed_avg = np.load('./loss & weights/loss_cicids_fedavg.npy')
    loss1 = np.load('loss_cicids_fedq_parallel.npy') if use_parallel else np.load('loss_cicids_fedq.npy')
    plot_loss(rounds, loss1, loss_fed_avg, use_parallel)
