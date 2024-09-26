import numpy as np
import models
import os
from tqdm import tqdm  # Import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from src.plotting import plotter1, plotter2, plotter3, plotter4

accuracy_per_class_over_rounds = []
precision_per_class_over_rounds = []
recall_per_class_over_rounds = []
f1_per_class_over_rounds = []
round_numbers = []
confusion_matrix_list = []

last_k_comparision = 3

from load_data import load_cicids_2017
X_train, X_valid, y_train, y_valid, smallest_k_ids = load_cicids_2017(last_k_comparision)


class Trainer:
    def __init__(self, dataset_x, dataset_y, epochs, learning_rate, batch_size):
        self.dataset_x = dataset_x
        self.dataset_y = dataset_y
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size

    def train(self, initial_weights):
        model = models.get_cicids_model()
        model.set_weights(initial_weights)
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = model.fit(self.dataset_x, self.dataset_y, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        output_weights = model.get_weights()
        final_loss = history.history['loss'][-1]
        final_accuracy = history.history['accuracy'][-1]
        return output_weights, final_loss, final_accuracy


def evaluate_global_model(global_model, global_weights, X_valid, y_valid):
    global_model.set_weights(global_weights)
    global_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    result = global_model.evaluate(X_valid, y_valid, verbose=0)  # Turn off logs here
    loss, accuracy = result[0], result[1]

    y_pred = np.argmax(global_model.predict(X_valid, verbose=0), axis=1)  # Turn off logs here

    cm = confusion_matrix(y_valid, y_pred)
    accuracy_per_class = cm.diagonal() / cm.sum(axis=1)

    precision_per_class = precision_score(y_valid, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_valid, y_pred, average=None)
    f1_per_class = f1_score(y_valid, y_pred, average=None)

    return loss, accuracy, accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, cm

def train_model(dataset_x, dataset_y, rounds, initial_weights):
    training_accuracy = []
    loss_list = []

    trainer = Trainer(dataset_x, dataset_y, epochs=2, learning_rate=0.01, batch_size=64)
    
    for round in range(1, rounds + 1):

        # Use tqdm for animated progress bar with `#` as the loading bar character
        with tqdm(total=100, desc=f"Training Round {round}", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}", ascii=True) as pbar:
            weights, loss, accuracy = trainer.train(initial_weights)
            
            # Update the progress bar
            pbar.update(100)
            pbar.set_postfix_str(f"accuracy = {accuracy:.4f}, loss = {loss:.4f}")

        training_accuracy.append(accuracy)
        loss_list.append(loss)

        # Update the weights for the next round of training
        initial_weights = weights

        valid_loss, valid_accuracy, accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, confusion_matrix = evaluate_global_model(global_model, weights, X_valid, y_valid)

        confusion_matrix_list.append(confusion_matrix)

        accuracy_per_class_over_rounds.append(accuracy_per_class)
        precision_per_class_over_rounds.append(precision_per_class)
        recall_per_class_over_rounds.append(recall_per_class)
        f1_per_class_over_rounds.append(f1_per_class)
        round_numbers.append(round)

        if round == rounds:
            np.savez('centrailsed_weights', *weights)

    return training_accuracy, loss_list


global_model = models.get_cicids_model()
global_weights = global_model.get_weights()
rounds = 3

# Train the model
training_accuracy, loss_list = train_model(X_train, y_train, rounds, global_weights)

# Print results
print("Training completed.")
print("Accuracies per round:", training_accuracy)
print("Losses per round:", loss_list)


experiment_name = "centrailsed"

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
# plotter3(cur_dir_path, valid_dist, experiment_name)
plotter4(cur_dir_path, smallest_k_ids, experiment_name)