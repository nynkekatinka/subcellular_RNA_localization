import torch
import numpy as np
from latent_statistics import permutation_test_cpu, permutation_test_gpu
from multiprocessing import Pool
import scanpy as sc
#from scipy.stats import ks_2samp
#import pandas as pd
import random
from dataprep import initialize_adata, subset_power_analysis
import pickle
import logging
import multiprocessing as mp

seed = 101
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
rng = np.random.RandomState(seed)

adata_split_cellID = sc.read_h5ad("/media/gambino/students_workdir/nynke/new_model_with_cell_id_left_out_custom_nynke_panel_simulated_embeddings_adata.h5ad")
adata_split_cellID = initialize_adata(adata_split_cellID)
adata_test = adata_split_cellID[adata_split_cellID.obs['cell_id'].isin(adata_split_cellID.uns['test_cellIDs'])]

# Set up logging
logging.basicConfig(filename='app.log', filemode='a', format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)


def compute_power_permutation(params):
    logging.info('Starting compute_power_permutation with params: %s', params)
    try:
        strength, count, sample = params
        significant_count = 0
        # Given that random patterns have only 1442 cells simulated, we don't calculate the power for these above 1400 so that we don't need to sample with replacement
        if count == '0-10' and sample > 1400:
            return (f'{strength}_{count}_{sample}', -1)
        if count == '10-30' and sample > 5100:
            return (f'{strength}_{count}_{sample}', -1)
        for i in range(1000):
            # sample new gene. No random seed so that every time a different "gene" is sampled.
            pattern = subset_power_analysis(adata_test, mixed_patterns = True, pattern_strength= strength, rna_count = count, sample_size = sample, random_seed=False)
            control = subset_power_analysis(adata_test, pattern = 'random', mixed_patterns = False, rna_count = count, sample_size = sample, random_seed=False)

            # Calculate null distribution and pvalue
            if sample > 1000:
                pvalue, observed_statistic, permuted_statistics = permutation_test_gpu(pattern.obsm["latent"], control.obsm["latent"], n_permutations=9999)
            else:
                pvalue, observed_statistic, permuted_statistics = permutation_test_cpu(pattern.obsm["latent"], control.obsm["latent"], n_permutations=9999)
            
            critical_value = 0.05
            if pvalue < critical_value:
                significant_count += 1
        result = (f'{strength}_{count}_{sample}', significant_count/1000)
        
        # Write the result to a file
        with open('/media/gambino/students_workdir/nynke/blurry/results.txt', 'a') as f:
            f.write(f'{strength}\t{count}\t{sample}\t{significant_count/1000}\n')
        logging.info('Finished compute_power_permutation with params: %s', params)

        return result
    except ValueError as e:
        if str(e) == "Cannot take a larger sample than population when 'replace=False'":
            print(f"Error for parameters: {params}")
            return None
        else:
            raise e


if __name__ == '__main__':
    strengths = ['strong', 'intermediate', 'low']
    counts = adata_test.obs['rna_count'].unique()
    samples = [5, 9, 15, 27, 46, 81, 142, 247, 432, 753, 1315, 2297, 4009, 7000]

    # Create a list of all combinations of strength, count, and sample
    combinations = [(strength, count, sample) for strength in strengths for count in counts for sample in samples]

    # Set the start method to 'spawn'
    mp.set_start_method('spawn', force=True)

    # Create a multiprocessing pool and compute the power for each combination
    with Pool(45) as p:
        results = p.map(compute_power_permutation, combinations)

    # in case want to do single processing:
    # results = [compute_power_permutation(combination) for combination in combinations]

    # Convert the results to a dictionary
    power_results = dict(results)

    path = "temp_objects/power_analysis_permutationLatent_to7000_logscale_uncorrected.pkl"

    # Open the file in write-binary mode and dump the object
    with open(path, 'wb') as f:
        pickle.dump(power_results, f)