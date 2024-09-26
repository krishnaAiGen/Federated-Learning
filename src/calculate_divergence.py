import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy.special import rel_entr

def compute_euclidean_distance(weights1, weights2):
    if isinstance(weights1, np.lib.npyio.NpzFile):
        weights1 = [weights1[f'arr_{i}'] for i in range(len(weights1))]
    
    if isinstance(weights2, np.lib.npyio.NpzFile):
        weights2 = [weights2[f'arr_{i}'] for i in range(len(weights2))]

    # Proceed with Euclidean distance calculation as before...
    flattened_weights1 = np.concatenate([w.flatten() for w in weights1])
    flattened_weights2 = np.concatenate([w.flatten() for w in weights2])
    
    return np.linalg.norm(flattened_weights1 - flattened_weights2)



def compute_cosine_similarity(weights1, weights2):
    if isinstance(weights1, np.lib.npyio.NpzFile):
        weights1 = [weights1[f'arr_{i}'] for i in range(len(weights1))]
    
    if isinstance(weights2, np.lib.npyio.NpzFile):
        weights2 = [weights2[f'arr_{i}'] for i in range(len(weights2))]

    # Flatten the weights to vectors for comparison
    flattened_weights1 = np.concatenate([w.flatten() for w in weights1])
    flattened_weights2 = np.concatenate([w.flatten() for w in weights2])
    
    # Compute cosine similarity
    cosine_similarity = dot(flattened_weights1, flattened_weights2) / (norm(flattened_weights1) * norm(flattened_weights2))
    return cosine_similarity


def compute_mean_absolute_difference(weights1, weights2):
    if isinstance(weights1, np.lib.npyio.NpzFile):
        weights1 = [weights1[f'arr_{i}'] for i in range(len(weights1))]
    
    if isinstance(weights2, np.lib.npyio.NpzFile):
        weights2 = [weights2[f'arr_{i}'] for i in range(len(weights2))]

    differences = [np.abs(w1 - w2).mean() for w1, w2 in zip(weights1, weights2)]
    return np.mean(differences)


def compute_kl_divergence(weights1, weights2):
    if isinstance(weights1, np.lib.npyio.NpzFile):
        weights1 = [weights1[f'arr_{i}'] for i in range(len(weights1))]
    
    if isinstance(weights2, np.lib.npyio.NpzFile):
        weights2 = [weights2[f'arr_{i}'] for i in range(len(weights2))]
        
    kl_divs = [np.sum(rel_entr(w1, w2)) for w1, w2 in zip(weights1, weights2)]
    return np.mean(kl_divs)
