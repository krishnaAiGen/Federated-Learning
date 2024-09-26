import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.special import rel_entr

import numpy as np

def compute_euclidean_distance(weights1, weights2):
    # Ensure both weights are lists of NumPy arrays
    if isinstance(weights1, str) or isinstance(weights2, str):
        raise ValueError("One of the weights sets is a string, expected arrays.")

    # Debug: Print the types of weights1 and weights2
    print(f"Type of weights1: {type(weights1)}, Type of weights2: {type(weights2)}")

    # Check and print the shape of the weights to make sure they are valid
    for i, weight in enumerate(weights1):
        if not hasattr(weight, 'shape'):
            raise ValueError(f"Weight {i+1} in weights1 is not an array but a {type(weight)}")
        print(f"Weight {i+1} shape: {weight.shape}")
    
    for i, weight in enumerate(weights2):
        if not hasattr(weight, 'shape'):
            raise ValueError(f"Weight {i+1} in weights2 is not an array but a {type(weight)}")
        print(f"Weight {i+1} shape: {weight.shape}")

    # Flatten and compute Euclidean distance between corresponding weights
    flattened_weights1 = np.concatenate([w.flatten() for w in weights1])
    flattened_weights2 = np.concatenate([w.flatten() for w in weights2])
    distance = np.linalg.norm(flattened_weights1 - flattened_weights2)
    
    return distance




def compute_cosine_similarity(weights1, weights2):
    # Flatten the weights to vectors for comparison
    flattened_weights1 = np.concatenate([w.flatten() for w in weights1])
    flattened_weights2 = np.concatenate([w.flatten() for w in weights2])
    
    # Compute cosine similarity
    cosine_similarity = dot(flattened_weights1, flattened_weights2) / (norm(flattened_weights1) * norm(flattened_weights2))
    return cosine_similarity


def compute_mean_absolute_difference(weights1, weights2):
    differences = [np.abs(w1 - w2).mean() for w1, w2 in zip(weights1, weights2)]
    return np.mean(differences)


def compute_kl_divergence(weights1, weights2):
    # Assume weights are probabilities (ensure non-zero values)
    kl_divs = [np.sum(rel_entr(w1, w2)) for w1, w2 in zip(weights1, weights2)]
    return np.mean(kl_divs)
