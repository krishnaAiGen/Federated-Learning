import numpy as np
import requests
from concurrent.futures import ThreadPoolExecutor
import models
from tqdm import tqdm  # Import tqdm

class Client:
    def __init__(self, dataset_x, dataset_y, epoch_number, learning_rate, batch):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.epoch_number = epoch_number
        self.learning_rate = learning_rate
        self.batch = batch

    def train(self, initial_weights):
        model = models.get_cicids_model()
        model.set_weights(initial_weights)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(self.dataset_x, self.dataset_y, epochs=self.epoch_number, batch_size=self.batch, verbose = 0)
        output_weights = model.get_weights()
        loss = history.history['loss'][-1]
        accuracy = history.history['accuracy'][-1]
        return output_weights, loss, accuracy

def train_server_without_parallelization(rounds, clients, global_weights, server_url):
    training_accuracy = []
    loss_list = []

    for round in range(1, rounds + 1):
        print(f"Training round {round}")
        client_weights = []

        # Use tqdm for animated progress bar with `#` as the loading bar character
        for i, client in enumerate(clients):
            with tqdm(total=100, desc=f"Client {i+1} training", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}", ascii=True) as pbar:
                weights, loss, accuracy = client.train(global_weights)
                client_weights.append(weights)

                # Update the progress bar (assuming each training is 100% done after each loop iteration)
                pbar.update(100)
                pbar.set_postfix_str(f"accuracy = {accuracy:.4f}, loss = {loss:.4f}")

        client_weights_list = [[w.tolist() for w in client] for client in client_weights]
        response = requests.post(f'{server_url}/update_weights', json={'weights': client_weights_list})
        updated_global = response.json()
        global_weights = [np.array(w) for w in updated_global['weights']]
        print(f"Performing federated averaging. Round = {updated_global['round']}, Accuracy = {updated_global['accuracy']}, Loss = {updated_global['loss']}")

        training_accuracy.append(updated_global['accuracy'])
        loss_list.append(updated_global['loss'])

    return training_accuracy, loss_list

def train_server_with_parallelization(rounds, clients, global_weights, server_url):
    training_accuracy = []
    loss_list = []

    def client_training(client, global_weights):
        return client.train(global_weights)

    for round in range(1, rounds + 1):
        print(f"Training round {round}")
        client_weights = []

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(client_training, client, global_weights) for client in clients]
            for future in futures:
                weights, loss, accuracy = future.result()
                client_weights.append(weights)
                print(f"Client training result: ############################# 100% accuracy = {accuracy}, loss = {loss}")

        client_weights_list = [[w.tolist() for w in client] for client in client_weights]
        response = requests.post(f'{server_url}/update_weights', json={'weights': client_weights_list})
        updated_global = response.json()
        global_weights = [np.array(w) for w in updated_global['weights']]
        print(f"Performing federated averaging. Round = {updated_global['round']}, Accuracy = {updated_global['accuracy']}, Loss = {updated_global['loss']}")

        training_accuracy.append(updated_global['accuracy'])
        loss_list.append(updated_global['loss'])

    return training_accuracy, loss_list
