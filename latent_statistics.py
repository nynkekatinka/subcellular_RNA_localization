import numpy as np
import random
from scipy.spatial.distance import cdist
from math import comb
import torch
import gc
import itertools

seed = 101
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def chamfer_L1_distance_cpu(point_cloud_1, point_cloud_2):
    # Compute all pairwise Manhattan distances, output matrix = [distance_cloudA1, distance_cloud2]
    distances_1_to_2 = cdist(point_cloud_1, point_cloud_2, metric='cityblock')
    distances_2_to_1 = cdist(point_cloud_2, point_cloud_1, metric='cityblock')

    # Get nearest neighbor for each point from point cloud 1 in point cloud 2 and vice versa
    distances_1_to_2 = np.min(distances_1_to_2, axis=1)
    distances_2_to_1 = np.min(distances_2_to_1, axis=1)

    # Compute the Chamfer distance
    return np.mean(distances_1_to_2) + np.mean(distances_2_to_1)


def permutation_test_cpu(pattern, control, n_permutations: int = 9999):
    observed_statistic = chamfer_L1_distance_cpu(pattern, control)
    combined = np.concatenate([pattern, control])
    len_combined = len(combined)
    num_pattern = len(pattern)

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
        permuted_statistic = chamfer_L1_distance_cpu(permuted_pattern, permuted_control)
        permuted_statistics[i] = permuted_statistic

    
    #These functions come from scipy.stats.permutation_test(). They have now been integrated in my main function in line to improve the efficiency
    eps =  (0 if not np.issubdtype(observed_statistic.dtype, np.inexact)
        else np.finfo(observed_statistic.dtype).eps*100)
    gamma = np.abs(eps * observed_statistic)
    #cmps_less = permuted_statistics <= observed_statistic + gamma
    cmps_greater = permuted_statistics >= observed_statistic - gamma
    # +1 is added to pvalues to add the observed value into the hypothetical population to make the pvalue more conservative. If it is an exact test, will use the true pvalue.
    adjustment = 0 if exact_test else 1
    #pvalues_less = (cmps_less.sum() + adjustment) / (n_permutations + adjustment)
    pvalues_greater = (cmps_greater.sum() + adjustment) / (n_permutations + adjustment)
    # I do a 1-tailed test because I only care if the observed statistic has a larger chamfer distance than the H0 population.
    p_value = pvalues_greater
    
    return p_value, observed_statistic, permuted_statistics



def chamfer_L1_distance_gpu(point_cloud_1, point_cloud_2):
    # Expand dims to prepare tensors for broadcasting (during pairwise distance calculation)
    point_cloud_1 = point_cloud_1.unsqueeze(1)
    point_cloud_2 = point_cloud_2.unsqueeze(0)

    # Compute all pairwise Manhattan distances. .sum(-1) -> sum distances of each latent dimension
    distances_1_to_2 = torch.abs(point_cloud_1 - point_cloud_2).sum(-1)
    distances_2_to_1 = torch.abs(point_cloud_2 - point_cloud_1).sum(-1)

    # Get nearest neighbor for each point from point cloud 1 in point cloud 2 and vice versa. 
    # Dim=1 & dim=0 because with the broadcasting we changed the tensor shapes to (n_points, n_points, n_dims)
    distances_1_to_2 = torch.min(distances_1_to_2, dim=1)[0]
    distances_2_to_1 = torch.min(distances_2_to_1, dim=0)[0]

    gc.collect()

    # Compute the Chamfer distance
    return torch.mean(distances_1_to_2) + torch.mean(distances_2_to_1)

def get_device():
    if torch.cuda.is_available():
        # Get the available devices (GPUs)
        devices = [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]

        # Get the available memory for each device
        mems = [torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device) for device in devices]

        # Select the device with the most available memory
        device = devices[mems.index(max(mems))]
    else:
        device = torch.device('cpu')

    return device

def permutation_test_gpu(pattern, control, n_permutations: int = 9999):
    combined = np.concatenate([pattern, control])
    len_combined = len(combined)
    num_pattern = len(pattern)

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
    device = get_device()
    pattern = torch.tensor(pattern).to(device)
    control = torch.tensor(control).to(device)
    combined = torch.tensor(combined).to(device)
    observed_statistic = chamfer_L1_distance_gpu(pattern, control)

    for i in range(n_permutations):
        permuted = combined[torch.randperm(len(combined))]
        permuted_pattern = permuted[:num_pattern]
        permuted_control = permuted[num_pattern:]
        permuted_statistic = chamfer_L1_distance_gpu(permuted_pattern, permuted_control)
        permuted_statistics[i] = permuted_statistic.item()
        gc.collect()
        torch.cuda.empty_cache()
    
    #pattern = pattern.cpu().numpy()
    #point_cloud_2 = point_cloud_2.cpu().numpy()
    # Convert observed_statistic to numpy array and get its data type
    #observed_statistic = observed_statistic.cpu().numpy()
    #permuted_statistics = permuted_statistics.cpu().numpy()

    #These functions come from scipy.stats.permutation_test(). They have now been integrated in my main function in line to improve the efficiency
    eps = 0 if not observed_statistic.is_floating_point() else torch.finfo(observed_statistic.dtype).eps*100
    gamma = torch.abs(eps * observed_statistic)
    permuted_statistics = torch.from_numpy(permuted_statistics).to(observed_statistic.device)
    #cmps_less = permuted_statistics <= observed_statistic + gamma
    cmps_greater = permuted_statistics >= observed_statistic - gamma
    # +1 is added to pvalues to add the observed value into the hypothetical population to make the pvalue more conservative. If it is an exact test, will use the true pvalue.
    adjustment = 0 if exact_test else 1
    #pvalues_less = (cmps_less.sum() + adjustment) / (n_permutations + adjustment)
    pvalues_greater = (cmps_greater.sum() + adjustment) / (n_permutations + adjustment)
    # I do a 1-tailed test because I only care if the observed statistic has a larger chamfer distance than the H0 population.
    p_value = pvalues_greater.cpu().numpy()
    
    
    # Delete variables to free GPU memory
    del pattern, control, combined, permuted, permuted_pattern, permuted_control, permuted_statistic, eps, gamma, cmps_greater, adjustment, pvalues_greater
    gc.collect()
    torch.cuda.empty_cache()
    
    return p_value, observed_statistic, permuted_statistics
