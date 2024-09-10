import numpy as np
import load_data
from collections import Counter

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) 


def find_q():
    X_train_list, y_train_list, X_valid, y_valid = load_data.load_cicids_2017_q()
    p_all_node_list = []
    
    # here malicious nodes are 3,5,7
    malicious_nodes = [3,5,7]
    for index in range(0, len(y_train_list)):
        counter_num = Counter(y_train_list[index])
        total_sample = len(y_train_list[index])
        try:
            benign_sample = counter_num[0]
        except:
            p_all_node_list.append(0)
        malicious_sample = total_sample - benign_sample
        p_node = malicious_sample/benign_sample
        p_all_node_list.append(p_node)
        
    q_all_list_node = softmax(p_all_node_list)
    return q_all_list_node


def get_coefficients(coeff_weights_list, client_accuracies, current_round):
    expo = True
    n = len(coeff_weights_list[0])
    if expo == False:
        coeff_weights = new_coeff_weights(n, client_accuracies, current_round, coeff_weights_list)
    
    if expo == True:
        coeff_weights = new_coeff_weights_expo(n, client_accuracies, current_round, coeff_weights_list)

    # scaler = MinMaxScaler()
    coeff_weights = np.array(coeff_weights)
    # coeff_weights = coeff_weights.reshape(-1, 1)
    # coeff_weights = scaler.fit_transform(coeff_weights)
    
    coeff_weights_norm = coeff_weights / coeff_weights.sum()

    return coeff_weights_norm



def sigmoid(delta_A, beta=1):
    return 1 / (1 + np.exp(-beta * delta_A))


def exponential_decay(delta_A, beta=1):
    return np.exp(-beta * delta_A)

def adaptive_reparam1(current_accuracy, previous_accuracy, current_coeff_weight, beta=0.5, alpha_min=0.5, alpha_max=1.0):
    th_accuracy = 0.9
    delta_accuracy = abs(th_accuracy - previous_accuracy)
    # delta_accuracy = current_accuracy - previous_accuracy
    # print(f"{current_accuracy:.4f}, {previous_accuracy:.4f}")
    
    # alpha1 = 1 - beta * delta_accuracy
    
    # alpha = np.clip(alpha1, alpha_min, alpha_max)

    
    # alpha = exponential_decay(delta_accuracy)
    alpha = sigmoid(delta_accuracy)

    alpha1 = alpha

    # Adjust the reparameterization weight
    new_coeff_weight = current_coeff_weight * alpha

    return new_coeff_weight, alpha1, alpha, delta_accuracy


def new_coeff_weights(n, client_accuracies, current_round, coeff_weights_list):
    new_weights = []

    for i in range(n):
        res, alpha1, alpha, delta_accuracy = adaptive_reparam1(client_accuracies[current_round-1][i], client_accuracies[current_round-2][i], coeff_weights_list[current_round-2][i])
        # print(f"{alpha1:.4f}, {alpha:.4f}, {delta_accuracy:.4f}")
        # print(res)
        new_weights.append(res)

    return new_weights


def adaptive_reparam1_expo(current_accuracy, previous_accuracy, current_coeff_weight):
    TRESHOLD = 0.90
    # lambda1 = -2
    # delta_accuracy = current_accuracy - TRESHOLD
    # exponent = -(lambda1 * delta_accuracy)
    # print("--------")
    # print(current_accuracy)
    # print(exponent)
    # print(pow(2, exponent))
    # new_coeff_weight = current_coeff_weight * pow(2, exponent)
    
    # print("calculation happening here")
    # print("old, new coefficient", current_coeff_weight, new_coeff_weight)
    # print("--------")

    print(abs(TRESHOLD - current_accuracy))
    return abs(TRESHOLD - current_accuracy)

def new_coeff_weights_expo(n, client_accuracies, current_round, coeff_weights_list):
    new_weights = []
    
    for i in range(n):
        res = adaptive_reparam1_expo(client_accuracies[current_round-1][i], client_accuracies[current_round-2][i], coeff_weights_list[current_round-2][i])

        new_weights.append(res)
    
    return new_weights