from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataprep import *
import logging
from math import comb
import numpy as np
import pickle
import random
import scanpy as sc
from scipy.spatial.distance import pdist, squareform
from scipy.stats import ks_2samp


def chamfer_L1_distance(distance_matrix, index_list):
    """ Returns the Manhattan based Chamfer-distance between all permutations of two point clouds by subsetting the distance matrix. 
    
    Inputs:
    distance_matrix: np.array of shape (n_cells, n_cells) representing the Manhattan distance between all cells in the combined pattern and control dataset.
    index_list: list of indices of two input genes that were randomly reassigned to either the test or control gene.
    """
    len_pattern = len(index_list) // 2
    distances_1_to_2 = np.min(distance_matrix[np.ix_(index_list[:len_pattern], index_list[len_pattern:])], axis=1)
    distances_2_to_1 = np.min(distance_matrix[np.ix_(index_list[len_pattern:], index_list[:len_pattern])], axis=1)
    return np.mean(distances_1_to_2) + np.mean(distances_2_to_1)

def latent_space_statistic(pattern, control, n_permutations: int = 9999, exact_test: bool = False, return_distances: bool = False):
    """ Computes a single instance of the Chamfer-based latent space statistic. 
    Computes the Manhattan distance between all pattern & control instances, and then uses this distance matrix to create a matrix with the Chamfer distance of each permutation. 
    It then calculates a one-sided p-value based on how many permutations are larger than the observed statistic. 
    
    Inputs:
    pattern: np.array of shape (n_cells, n_latent) representing the latent space of the test gene. Subset of adata.obsm['latent'].
    control: np.array of shape (n_cells, n_latent) representing the latent space of the control. Subset of adata.obsm['latent'].
    n_permutations: int, number of permutations to perform. Default is 9999.
    exact_test: bool, whether to perform an exact test. Default is False. True if n_permutation is smaller than default n_permutations.
    return_distances: bool, whether to return the distances of the permutations. Default is False.
    """

    # Compute Manhattan distance between all pattern and control instances. 
    combined = np.concatenate([pattern, control])
    distance_matrix = squareform(pdist(combined, metric='cityblock'))

    # Compute observed chamfer distance 
    len_combined = len(combined)
    observed_statistic = chamfer_L1_distance(distance_matrix, list(range(len_combined)))

    # Generate random permutations of the indices to subset the distance matrix
    index_lists = np.apply_along_axis(np.random.permutation, 1, np.tile(list(range(len_combined)), (n_permutations, 1)))

    # Compute chamfer distances for all permutations
    chamfer_distances = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(chamfer_L1_distance, distance_matrix, index_list) for index_list in index_lists]
        for future in as_completed(futures):
            chamfer_distances.append(future.result())
    chamfer_distances = np.array(chamfer_distances)

    # Compute p-value, based on permutation test from scipy.stats: number of permutations larger than observed statistic divided by total number of permutations. 
    # Due to the use of finite precision arithmetic, numerically distinct values could be returned whereas theoretical values would be exactly equal. 
    # Therefore, we use a gamma adjustment to also consider elements in the null distribution that are "close" (within a relative tolerance of 100 times the floating point epsilon of inexact dtypes) to the observed value of the test statistic as equal. 
    eps = (0 if not np.issubdtype(observed_statistic.dtype, np.inexact)
           else np.finfo(observed_statistic.dtype).eps * 100)
    gamma = np.abs(eps * observed_statistic)
    cmps_greater = chamfer_distances >= observed_statistic - gamma
    adjustment = 0 if exact_test else 1

    # one sided p-value, because we do not care if the observed statistic is significantly smaller than the null distribution.
    pvalues_greater = (cmps_greater.sum() + adjustment) / (n_permutations + adjustment)
    p_value = pvalues_greater

    if return_distances:
        return p_value, observed_statistic, chamfer_distances
    else:
        return p_value

def process_sample(i, adata_test, strength, rna_count, cell_count, n_permutations, exact_test):
        pattern = create_simulated_gene(adata_test, mixed_patterns=True, pattern_strength=strength, rna_count=rna_count, sample_size=cell_count, random_seed=False)
        control = create_simulated_gene(adata_test, pattern='random', mixed_patterns=False, rna_count=rna_count, sample_size=cell_count, random_seed=False)
        return latent_space_statistic(pattern.obsm["latent"], control.obsm["latent"], n_permutations, exact_test=exact_test, return_distances=False)

def compute_power_latent_space(params, adata_test):
    logging.info('Starting compute_power_latent_space with params: %s', params)
    strength, rna_count, cell_count = params

    # Given that random patterns have only 1442 cells simulated, we don't calculate the power for these above 1400 so that we don't need to sample with replacement
    if (rna_count == '0-10' and cell_count > 1400) or (rna_count == '10-30' and cell_count > 5100):
        with open('/media/gambino/students_workdir/nynke/blurry/results_permutation.txt', 'a') as f:
            f.write(f'{strength}\t{rna_count}\t{cell_count}\t-1\t-1\n')
        return (f'{strength}_{rna_count}_{cell_count}', -1)   # to easily filter out later.
    
    # Count max number of permutations with Combination rule nCr, where r is the pattern size
    if cell_count < 15:
        total_permutations = comb(cell_count*2, cell_count) # built in implementation of nCr rule.
        # Adjust n_permutations if it's larger than total_permutations
        if 9999 > total_permutations:
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
        with ProcessPoolExecutor(max_workers=8) as executor: # max_workers=4 , max_workers=8 at night
            futures = [executor.submit(process_sample, i, adata_test, strength, rna_count, cell_count, n_permutations, exact_test) for i in range(1000)]
            pvalues = [future.result() for future in as_completed(futures)]
            
        pvalues = np.array(pvalues)
        critical_value = 0.05
        adjusted_critical_value = critical_value / 5000
        significant_count = np.sum(pvalues < critical_value)
        bonferroni_count = np.sum(pvalues < adjusted_critical_value)

        with open('/media/gambino/students_workdir/nynke/blurry/results_permutation.txt', 'a') as f:
            f.write(f'{strength}\t{rna_count}\t{cell_count}\t{significant_count / 1000}\t{bonferroni_count / 1000}\n')

        result = (f'{strength}_{rna_count}_{cell_count}', significant_count/1000)
        bonferroni_result = (f'{strength}_{rna_count}_{cell_count}', bonferroni_count/1000)

        logging.info('Finished compute_power_permutation with params: %s', params)

        return result, bonferroni_result
    except ValueError as e:
        if str(e) == "Cannot take a larger sample than population when 'replace=False'":
            print(f"Error for parameters: {params}")
            return None
        else:
            raise e

def compute_power_rf(params, rf_singlejob, adata_test):
    """
    Function to compute the power of the RF classifier for a given parameter combination.
    
    inputs:
    params: tuple of pattern strength, RNA count, cell count
    rf_singlejob: RF classifier object. Should be created as a single job object.
    adata_test: AnnData object containing the test dataset.
    
    output:
    result, bonferroni_result: tuple of parameter combination and power, tuple of parameter combination and Bonferroni corrected power
    """
    logging.info('Starting compute_power_rf with params: %s', params)
    try:
        strength, rna_count, cell_count = params
        significant_count = 0
        bonferroni_count = 0

        # Given that random patterns have only 1442 cells simulated, we don't calculate the power for these above 1400 so that we don't need to sample with replacement
        if (rna_count == '0-10' and cell_count > 1400) or (rna_count == '10-30' and cell_count > 5100):
            with open('/media/gambino/students_workdir/nynke/blurry/power_rf.txt', 'a') as f:
                f.write(f'{strength}\t{rna_count}\t{cell_count}\t-1\t-1\n')
            return (f'{strength}_{rna_count}_{cell_count}', -1)
        
        for i in range(1000):
            # sample new gene. No random seed so that every time a different "gene" is sampled.
            pattern = create_simulated_gene(adata_test, mixed_patterns = True, pattern_strength= strength, rna_count = rna_count, sample_size = cell_count)
            control = create_simulated_gene(adata_test, pattern = 'random', mixed_patterns = False, rna_count = rna_count, sample_size = cell_count)

            # Calculate RF classification probabilities
            pattern_score = rf_singlejob.predict_proba(pattern.obsm["latent"])[:,1]
            control_score = rf_singlejob.predict_proba(control.obsm["latent"])[:,1]

            statistic, pvalue = ks_2samp(pattern_score, control_score)
            critical_value = 0.05
            adjusted_critical_value = critical_value / 5000
            if pvalue < critical_value:
                significant_count += 1
            if pvalue < adjusted_critical_value:
                bonferroni_count += 1
        
        with open('/media/gambino/students_workdir/nynke/blurry/power_rf.txt', 'a') as f:
            f.write(f'{strength}\t{rna_count}\t{cell_count}\t{significant_count / 1000}\t{bonferroni_count / 1000}\n')

        result = (f'{strength}_{rna_count}_{cell_count}', significant_count/1000)
        bonferroni_result = (f'{strength}_{rna_count}_{cell_count}', bonferroni_count/1000)

        logging.info('Finished compute_power_rf with params: %s', params)

        return result, bonferroni_result
    
    except ValueError as e:
        if str(e) == "Cannot take a larger sample than population when 'replace=False'":
            print(f"Error for parameters: {params}")
            return None
        else:
            raise e

if __name__ == '__main__':
    """ Main function to run the power analysis for the permutation test, currently set up for the latent space method (i.e. Chamfer distance).
    If you want to run the RF method, load the RF classifier object here, which should be trained as a single job object as we multiprocess the power analysis.
    
    Creates a dictionary with the power results for each parameter combination and saves it to a pickle file. Within the functions, the results are also saved to a text file. For further visualizations, see the notebook 'Power Analysis'.
    """
    # Set up logging
    logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

    adata_split_cellID = sc.read_h5ad("/media/gambino/students_workdir/nynke/new_model_with_cell_id_left_out_custom_nynke_panel_simulated_embeddings_adata.h5ad")
    adata_split_cellID = initialize_adata(adata_split_cellID)
    adata_test = adata_split_cellID[adata_split_cellID.obs['cell_id'].isin(adata_split_cellID.uns['test_cellIDs'])]

    logging.info('Loaded the dataset')

    strengths = ['strong', 'intermediate', 'low'] 
    rna_counts = adata_test.obs['rna_count'].unique()

    # Generated 14 points between 5 and 7000 on a log scale:
    # points = np.logspace(np.log10(5), np.log10(7000), num=14)
    # cell_counts = np.round(points).astype(int) # output is separated by /t iso commas, so can't use directly in the code

    cell_counts = [5, 9, 15, 27, 46, 81, 142, 247, 432, 753, 1315, 2297, 4009, 7000]

    # Create a list of all combinations of strength, count, and cell_count
    combinations = [(strength, rna_count, cell_count) for strength in strengths for rna_count in rna_counts for cell_count in cell_counts]

    logging.info('Starting each param combination...')

    results = [compute_power_latent_space(params, adata_test) for params in combinations]

    # Convert the results to a dictionary
    power_results = {result[0]: result[1] for result, _ in results if result}
    bonferroni_power_results = {bonferroni_result[0]: bonferroni_result[1] for _, bonferroni_result in results if bonferroni_result}

    path = "temp_objects/power_analysis/latent_space_logscale_uncorrected.pkl"
    bonferroni_path = "temp_objects/power_analysis/latent_space_logscale_bonferroni.pkl"

    # Open the file in write-binary mode and dump the object
    with open(path, 'wb') as f:
        pickle.dump(power_results, f)

    # Open the file in write-binary mode and dump the object
    with open(bonferroni_path, 'wb') as f:
        pickle.dump(bonferroni_power_results, f)
