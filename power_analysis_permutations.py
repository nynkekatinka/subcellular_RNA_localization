import random
from math import comb
from dataprep import *
import numpy as np
from scipy.spatial.distance import pdist, squareform
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import logging
import pickle
import scanpy as sc

def chamfer_L1_distance(distance_matrix, index_list):
    len_pattern = len(index_list) // 2
    distances_1_to_2 = np.min(distance_matrix[np.ix_(index_list[:len_pattern], index_list[len_pattern:])], axis=1)
    distances_2_to_1 = np.min(distance_matrix[np.ix_(index_list[len_pattern:], index_list[:len_pattern])], axis=1)
    return np.mean(distances_1_to_2) + np.mean(distances_2_to_1)

def test_permutation(pattern, control, n_permutations: int = 9999, exact_test: bool = False, return_distances: bool = False):
    ''' pattern and control are subsets of adata.obsm['latent'].
    '''
    combined = np.concatenate([pattern, control])
    distance_matrix = squareform(pdist(combined, metric='cityblock'))
    len_combined = len(combined)
    num_pattern = len(pattern)
    observed_statistic = chamfer_L1_distance(distance_matrix, list(range(len_combined)))

    index_lists = np.apply_along_axis(np.random.permutation, 1, np.tile(list(range(len_combined)), (n_permutations, 1)))

    chamfer_distances = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(chamfer_L1_distance, distance_matrix, index_list) for index_list in index_lists]
        for future in as_completed(futures):
            chamfer_distances.append(future.result())

    chamfer_distances = np.array(chamfer_distances)

    eps = (0 if not np.issubdtype(observed_statistic.dtype, np.inexact)
           else np.finfo(observed_statistic.dtype).eps * 100)
    gamma = np.abs(eps * observed_statistic)
    cmps_greater = chamfer_distances >= observed_statistic - gamma
    adjustment = 0 if exact_test else 1
    pvalues_greater = (cmps_greater.sum() + adjustment) / (n_permutations + adjustment)
    p_value = pvalues_greater

    if return_distances:
        return p_value, observed_statistic, chamfer_distances
    else:
        return p_value

def process_sample(i, adata_test, strength, count, sample, exact_test):
        pattern = subset_power_analysis(adata_test, mixed_patterns=True, pattern_strength=strength, rna_count=count, sample_size=sample, random_seed=False)
        control = subset_power_analysis(adata_test, pattern='random', mixed_patterns=False, rna_count=count, sample_size=sample, random_seed=False)
        return test_permutation(pattern.obsm["latent"], control.obsm["latent"], n_permutations=9999, exact_test=exact_test, return_distances=False)

def compute_power_permutation(params, adata_test):
    logging.info('Starting compute_power_permutation with params: %s', params)
    strength, count, sample = params

    # Given that random patterns have only 1442 cells simulated, we don't calculate the power for these above 1400 so that we don't need to sample with replacement
    if (count == '0-10' and sample > 1400) or (count == '10-30' and sample > 5100):
        with open('/media/gambino/students_workdir/nynke/blurry/results.txt', 'a') as f:
            f.write(f'{strength}\t{count}\t{sample}\t-1\t-1\n')
        return (f'{strength}_{count}_{sample}', -1)
    
    # Count max number of permutations with Combination rule nCr, where r is the pattern size
    if sample < 15:
        total_permutations = comb(sample*2, sample) # built in implementation of nCr rule.
        # Adjust n_permutations if it's larger than total_permutations
        if n_permutations > total_permutations:
            exact_test = True
            n_permutations = int(total_permutations)
        else:
            n_permutations = 9999
            exact_test = False
    else:
        # If num_pattern is 15, the total combinations are 1.5e8, which already is much larger than 9999. So we skip calculating the factorials for 15+ to save compute time. 
        n_permutations = 9999
        exact_test = False

    try:
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_sample, i, adata_test, strength, count, sample, exact_test) for i in range(1000)]
            pvalues = [future.result() for future in as_completed(futures)]
            
        pvalues = np.array(pvalues)
        critical_value = 0.05
        adjusted_critical_value = critical_value / 5000
        significant_count = np.sum(pvalues < critical_value)
        bonferroni_count = np.sum(pvalues < adjusted_critical_value)

        with open('/media/gambino/students_workdir/nynke/blurry/results.txt', 'a') as f:
            f.write(f'{strength}\t{count}\t{sample}\t{significant_count / 1000}\t{bonferroni_count / 1000}\n')

        result = (f'{strength}_{count}_{sample}', significant_count/1000)
        bonferroni_result = (f'{strength}_{count}_{sample}', bonferroni_count/1000)

        logging.info('Finished compute_power_permutation with params: %s', params)

        return result, bonferroni_result
    except ValueError as e:
        if str(e) == "Cannot take a larger sample than population when 'replace=False'":
            print(f"Error for parameters: {params}")
            return None
        else:
            raise e

if __name__ == '__main__':
    # Set up logging
    logging.basicConfig(filename='permute.log', filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    adata_split_cellID = sc.read_h5ad("/media/gambino/students_workdir/nynke/new_model_with_cell_id_left_out_custom_nynke_panel_simulated_embeddings_adata.h5ad")
    adata_split_cellID = initialize_adata(adata_split_cellID)
    adata_test = adata_split_cellID[adata_split_cellID.obs['cell_id'].isin(adata_split_cellID.uns['test_cellIDs'])]

    logging.info('Loaded the dataset')

    strengths = ['strong', 'intermediate', 'low']
    counts = adata_test.obs['rna_count'].unique()
    #samples = [5, 9, 15, 27, 46, 81, 142, 247, 432, 753, 1315, 2297, 4009, 7000]
    samples = [1315, 2297, 4009] # , 7000]

    # Create a list of all combinations of strength, count, and sample
    combinations = [(strength, count, sample) for strength in strengths for count in counts for sample in samples]
    combinations.append(('low', '0-10', 753))

    logging.info('Starting each param combination...')

    results = [compute_power_permutation(params, adata_test) for params in combinations]

    # Convert the results to a dictionary
    power_results = {result[0]: result[1] for result, _ in results if result}
    bonferroni_power_results = {bonferroni_result[0]: bonferroni_result[1] for _, bonferroni_result in results if bonferroni_result}

    path = "temp_objects/power_analysis_permutationLatent_to7000_logscale_uncorrected.pkl"
    bonferroni_path = "temp_objects/power_analysis_permutationLatent_to7000_logscale_bonferroni.pkl"

    # Open the file in write-binary mode and dump the object
    with open(path, 'wb') as f:
        pickle.dump(power_results, f)

    # Open the file in write-binary mode and dump the object
    with open(bonferroni_path, 'wb') as f:
        pickle.dump(bonferroni_power_results, f)
