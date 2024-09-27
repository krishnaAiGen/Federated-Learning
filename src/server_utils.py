from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import numpy as np
import models

def initialize_global_model(initialise_new, save):
    global_model = models.get_cicids_model()
        
    if initialise_new:
        global_weights = global_model.get_weights()
        if save:
            np.savez('initial_global_weights', *global_weights)
    else:
        loaded_weights = np.load('initial_global_weights.npz', allow_pickle=True)
        global_weights = [loaded_weights[f'arr_{i}'] for i in range(len(loaded_weights))]
        global_model.set_weights(global_weights)

    return global_model, global_weights

def weights_to_list(weights):
    return [w.tolist() for w in weights]

# def weights_to_list(weights):
#     return [w.numpy().tolist() for w in weights]


def list_to_weights(weights_list):
    return [np.array(w) for w in weights_list]

def evaluate_global_model(global_model, global_weights, X_valid, y_valid):
    global_model.set_weights(global_weights)
    global_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    result = global_model.evaluate(X_valid, y_valid, verbose=0)  # Turn off logs here
    loss, accuracy = result[0], result[1]

    y_pred = np.argmax(global_model.predict(X_valid, verbose=0), axis=1)  # Turn off logs here

    cm = confusion_matrix(y_valid, y_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]) 
    accuracy_per_class = np.divide(cm.diagonal(), cm.sum(axis=1), where=cm.sum(axis=1)!=0)

    precision_per_class = precision_score(y_valid, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_valid, y_pred, average=None, zero_division=1)
    f1_per_class = f1_score(y_valid, y_pred, average=None)

    return loss, accuracy, accuracy_per_class, precision_per_class, recall_per_class, f1_per_class, cm
