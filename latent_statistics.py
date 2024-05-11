import numpy as np
import random
from scipy.spatial.distance import cdist


seed = 101
random.seed(seed)
np.random.seed(seed)

def chamfer_L1_distance(point_cloud_1, point_cloud_2):

    # Compute all pairwise Manhattan distances, output matrix = [distance_cloudA1, distance_cloud2]
    distances_1_to_2 = cdist(point_cloud_1, point_cloud_2, metric='cityblock')
    distances_2_to_1 = cdist(point_cloud_2, point_cloud_1, metric='cityblock')

    # Get nearest neighbor for each point from point cloud 1 in point cloud 2 and vice versa
    distances_1_to_2 = np.min(distances_1_to_2, axis=1)
    distances_2_to_1 = np.min(distances_2_to_1, axis=1)

    # Compute the Chamfer distance
    return np.mean(distances_1_to_2) + np.mean(distances_2_to_1)

def permutation_test(pattern, control, n_permutations: int = 1000):
    observed_statistic = chamfer_L1_distance(pattern, control)
    num_pattern = len(pattern)
    combined = np.concatenate([pattern, control])
    
    # Preallocate permuted null hypothesis array
    permuted_statistics = np.empty(n_permutations)

    for i in range(n_permutations):
        permuted = np.random.permutation(combined)
        permuted_pattern = permuted[:num_pattern]
        permuted_control = permuted[num_pattern:]
        permuted_statistic = chamfer_L1_distance(permuted_pattern, permuted_control)
        permuted_statistics[i] = permuted_statistic
    
    #These functions come from scipy.stats.permutation_test(). They have now been integrated in my main function in line to improve the efficiency
    eps =  (0 if not np.issubdtype(observed_statistic.dtype, np.inexact)
            else np.finfo(observed_statistic.dtype).eps*100)
    gamma = np.abs(eps * observed_statistic)
    cmps_less = permuted_statistics <= observed_statistic + gamma
    cmps_greater = permuted_statistics >= observed_statistic - gamma
    # +1 is added to pvalues to add the observed value into the hypothetical population to make the pvalue more conservative.
    pvalues_less = (cmps_less.sum() + 1) / (n_permutations + 1)
    pvalues_greater = (cmps_greater.sum() + 1) / (n_permutations + 1)
    #because with 2-tailed you should use alpha=0.025 as treshold, so now it gets scaled back to 0.05
    p_value = np.minimum(pvalues_less, pvalues_greater) * 2 

    return p_value, observed_statistic, permuted_statistics