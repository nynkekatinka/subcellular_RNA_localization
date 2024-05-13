import numpy as np
import random
from scipy.spatial.distance import cdist
from math import factorial, comb
from multiprocessing import Pool

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


def permutation_test(pattern, control, n_permutations: int = 9999):
    observed_statistic = chamfer_L1_distance(pattern, control)
    num_pattern = len(pattern)
    combined = np.concatenate([pattern, control])
    len_combined = len(combined)

    # Count max number of permutations with Combination rule nCr, where r is the pattern size
    if num_pattern < 15:
        total_permutations = comb(len_combined, num_pattern) # built in implementation of nCr rule.

        # Adjust n_permutations if it's larger than total_permutations
        if n_permutations > total_permutations:
            exact_test = True
            n_permutations = int(total_permutations)
        else:
            exact_test = False
    else:
        # If num_pattern is 15, the total combinations are 1.5e8, which already is much larger than 9999. So we skip calculating the factorials for 15+ to save compute time. 
        exact_test = False

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
    # +1 is added to pvalues to add the observed value into the hypothetical population to make the pvalue more conservative. If it is an exact test, will use the true pvalue.
    adjustment = 0 if exact_test else 1
    pvalues_less = (cmps_less.sum() + adjustment) / (n_permutations + adjustment)
    pvalues_greater = (cmps_greater.sum() + adjustment) / (n_permutations + adjustment)
    #because with 2-tailed you should use alpha=0.025 as treshold, so now it gets scaled back to 0.05
    p_value = np.minimum(pvalues_less, pvalues_greater) * 2 

    return p_value, observed_statistic, permuted_statistics
