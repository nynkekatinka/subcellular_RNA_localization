import pandas as pd
import numpy as np
import sklearn.model_selection as skm
import anndata as ad

seed = 101

def map_interval(interval):
    if interval == '0-10':
        return '0-10'
    elif interval in ['10-20', '20-30']:
        return '10-30'
    elif interval in ['30-40', '40-50', '50-60']:
        return '30-60'
    elif interval in ['70-80', '80-90', '90-100']:
        return '70-100'
    elif interval == '100+':
        return '100+'
    else:
        return None
    
def initialize_adata(adata):
    choices = ['strong', 'intermediate', 'low']
    conditions = [
        (adata.obs['prop'] == 0.9) | ((adata.obs['prop'] == 0.4) & (adata.obs['pattern'] == 'protrusion')),
        (adata.obs['prop'] == 0.5) | ((adata.obs['prop'] == 0.2) & (adata.obs['pattern'] == 'protrusion')),
        (adata.obs['prop'] == 0.1) | ((adata.obs['prop'] == 0.0) & (adata.obs['pattern'] == 'protrusion'))
    ]
    adata.obs['pattern_strength'] = np.select(conditions, choices, default='unknown')

    # Include modified RNA count intervals in the adata object
    adata.obs['rna_count'] = adata.obs['n_spots_interval'].apply(map_interval)


    return adata

def train_val_test(adata, seed: int, split_per_cellID: bool = True):
    """ 
    Split the anndata object into train, validation and test anndata objects with an 80-10-10 split.
    Can choose to split with or without maintaining the grouping of cell_ids.
    """

    all_indices = adata.obs.index

    if split_per_cellID == True:
        cell_ids = np.array(adata.obs['cell_id'])
        unique_cell_ids, cell_id_indices = np.unique(cell_ids, return_inverse=True)

        # Perform train-val-test split on cell_ids, maintaining the grouping
        train_indices, test_val_indices = skm.train_test_split(np.arange(len(unique_cell_ids)), test_size=0.2, random_state=seed)
        val_indices, test_indices = skm.train_test_split(test_val_indices, test_size=0.5, random_state=seed)

        # Map the grouped indices back to the original indices
        train_indices = np.concatenate([np.where(cell_id_indices == i)[0] for i in train_indices])
        val_indices = np.concatenate([np.where(cell_id_indices == i)[0] for i in val_indices])
        test_indices = np.concatenate([np.where(cell_id_indices == i)[0] for i in test_indices])
    else:
        train_indices, test_val_indices = skm.train_test_split(all_indices, test_size=0.2, random_state=seed)
        val_indices, test_indices = skm.train_test_split(test_val_indices, test_size=0.5, random_state=seed)

    # Subset AnnData object using the obtained indices
    adata_train = adata[train_indices, :]
    adata_val = adata[val_indices, :]
    adata_test = adata[test_indices, :]

    return adata_train, adata_val, adata_test


def train_test(adata, seed: int, split_per_cellID: bool = True):
    """ 
    Split the anndata object into train and test ad.AnnData objects with an 80-20 split. 
    Can choose to split with or without maintaining the grouping of cell_ids.
    """
    all_indices = adata.obs.index

    if split_per_cellID == True:
        cell_ids = np.array(adata.obs['cell_id'])
        unique_cell_ids, cell_id_indices = np.unique(cell_ids, return_inverse=True)

        # Perform train-val-test split on cell_ids, maintaining the grouping
        train_indices, test_indices = skm.train_test_split(np.arange(len(unique_cell_ids)), test_size=0.2, random_state=seed)

        # Map the grouped indices back to the original indices
        train_indices = np.concatenate([np.where(cell_id_indices == i)[0] for i in train_indices])
        test_indices = np.concatenate([np.where(cell_id_indices == i)[0] for i in test_indices])
    else:
        train_indices, test_indices = skm.train_test_split(all_indices, test_size=0.2, random_state=seed)

    # Subset AnnData object using the obtained indices
    adata_train = adata[train_indices, :]
    adata_test = adata[test_indices, :]

    return adata_train, adata_test

# TO DO, can make mixed patterns including or excluding random. For now will do it excluding random. 
def subsetGenes_splitCellID(adata, pattern: str = 'pericellular', mixed_patterns: bool = False, pattern_strength: str = "strong", rna_count: str = '10-30', mixed_counts: bool = False, random_seed: int = 101):
    """
    Subset the anndata object into a `1 gene multiple cells` object. Can filter the cells based on the number of spots, the pattern and the pattern strength.

    Parameters
    ----------
    adata : ad.AnnData object
        complete anndata object.
    pattern : str
        Type of subcellular expression pattern you want to filter on. Default is 'pericellular', which has the highest f1 RF score.
    mixed_patterns: bool
        True: all patterns can be included, False: only the pattern type specified in `pattern` is included. Default is False. 
    pattern_strength : str
        strength of the pattern, which is labeled as pattern_strength in the anndata object. Strong, which comes down to 45% of points fall in the pattern for protrusion, 90% of points are in pattern for all the other patterns. 
        If the pattern is random, then pattern_strength is not used, given that irrelevant for random when using the simFISH v2 definition of patterness (90% of points assigned to the pattern 'random' will still amount to 100% randomness).
    high_or_low : str
        Whether you want to filter genes with a higher or lower count than the given threshold. Default is lower.
        If none, then no threshold is chosen and mixed counts are included. 
    count_threshold : int
        Count threshold to filter on. Default is 11, so that genes with count 0-10 are selected. For high, the threshold is 140.
    mixed_counts: bool
        True: all counts are included, False: only counts above or below the threshold are included. Default is False.
        
    Returns
    -------
    ad.AnnData

    """
    adata_filtered = adata[(
                                adata.obs['pattern'] == pattern if mixed_patterns == False
                                else adata.obs['pattern'] != 'random'
                            ) & 
                           (
                                adata.obs['rna_count'] == rna_count if mixed_counts == False 
                                else True
                            ) & 
                           (
                               adata.obs['pattern_strength'] == pattern_strength if pattern != 'random' 
                               else True
                            )
                        ].copy()

    
    subset_dict = {}
    i = 0
    while len(adata_filtered.obs['cell_id'].unique()) > 0:
        grouped = adata_filtered.obs.groupby('cell_id')

        # For each group, select at most one item & concatenate results back into a DataFrame
        subsets = [group.sample(n=1, random_state = random_seed) for _, group in grouped]
        subset_obs = pd.concat(subsets)

        # Use this DataFrame to subset the original anndata object and store in dictionary, while omitting the selected cells from the original adata object
        count_subset = adata_filtered[adata_filtered.obs.index.isin(subset_obs.index)].copy()
        subset_dict[i] = count_subset
        adata_filtered = adata_filtered[~adata_filtered.obs.index.isin(subset_obs.index)].copy()
        i += 1

    return subset_dict

def subset_power_analysis(adata, pattern: str = 'pericellular', mixed_patterns: bool = True, pattern_strength: str = "strong", rna_count: str = '10-30', sample_size: int = 5, random_seed: bool = False):
    """
    Subset the anndata object into a `1 gene multiple cells` object. Can filter the cells based on the number of spots, the pattern and the pattern strength.

    Parameters
    ----------
    adata : ad.AnnData object
        complete anndata object.
    pattern : str
        Type of subcellular expression pattern you want to filter on. Default is 'pericellular', which has the highest f1 RF score.
    mixed_patterns: bool
        True: all patterns can be included, False: only the pattern type specified in `pattern` is included. Default is False. 
    pattern_strength : str
        strength of the pattern, which is labeled as pattern_strength in the anndata object. Strong, which comes down to 45% of points fall in the pattern for protrusion, 90% of points are in pattern for all the other patterns. 
        If the pattern is random, then pattern_strength is not used, given that irrelevant for random when using the simFISH v2 definition of patterness (90% of points assigned to the pattern 'random' will still amount to 100% randomness).
    high_or_low : str
        Whether you want to filter genes with a higher or lower count than the given threshold. Default is lower.
        If none, then no threshold is chosen and mixed counts are included. 
    count_threshold : int
        Count threshold to filter on. Default is 11, so that genes with count 0-10 are selected. For high, the threshold is 140.
    mixed_counts: bool
        True: all counts are included, False: only counts above or below the threshold are included. Default is False.
        
    Returns
    -------
    ad.AnnData

    """
    adata_filtered = adata[(
                                (
                                    adata.obs['pattern'] == pattern if mixed_patterns == False
                                    else adata.obs['pattern'] != 'random'
                                )
                            ) & 
                           (
                                adata.obs['rna_count'] == rna_count
                            ) & 
                           (
                               adata.obs['pattern_strength'] == pattern_strength if pattern != 'random' 
                               else True
                            )
                        ].copy()

    if random_seed == True:
        subset = adata_filtered.obs.sample(n=sample_size, random_state = seed) 
    else:
        subset = adata_filtered.obs.sample(n=sample_size)
    
    adata_subset = adata_filtered[adata_filtered.obs.index.isin(subset.index)]


    return adata_subset


def balanceTrainingData_pattern_noPattern(adata, random_seed: int = 101):
    """
    Downsample the anndata object so that the test and control group for RF are equal size, while the control group is balanced for patterns and spot count. 
    Assumes there are less test observations than the other subcellular expression patterns combined. 

    Parameters
    ----------
    adata : ad.AnnData object
        complete anndata object.
    random_seed: int
        Seed for reproducability purposes. 
        
    Returns
    -------
    balanced_adata : ad.AnnData
        test and control group are equal size, while the control group is balanced for patterns and spot count
    """
        
    adata_test = adata[adata.obs['random_or_pattern']=='pattern']
    adata_control = adata[adata.obs['random_or_pattern']=='no_pattern']
    
    sample_size = len(adata_control)
    
    subset = adata_test.obs['n_spots_interval'].sample(n=sample_size, random_state=random_seed)
    
    adata_test_subset = adata_test[adata_test.obs.index.isin(subset.index)]
    
    # Concatenate adata_test and adata_control_subset
    adata = ad.concat([adata_test_subset, adata_control])

    return adata


def balanceTrainingData(adata, testPattern: str = 'pericellular', include_random: bool = False, random_seed: int = 101):
    """
    Downsample the anndata object so that the test and control group for RF are equal size, while the control group is balanced for patterns and spot count. 
    Assumes there are less test observations than the other subcellular expression patterns combined. 

    Parameters
    ----------
    adata : ad.AnnData object
        complete anndata object.
    testPattern : str
        Type of subcellular expression pattern you want to use as the test case. All other patterns will be used as control. 
    pattern_no_pattern: bool
        True: test pattern or no pattern, False: test patterns versus other patterns. Default is False.
    include_random: bool
        Whether the `no_pattern` pattern is included in the data or not. Default is False. 
    random_seed: int
        Seed for reproducability purposes. 
        
    Returns
    -------
    balanced_adata : ad.AnnData
        test and control group are equal size, while the control group is balanced for patterns and spot count
    """

    if include_random == False:
        adata = adata[adata.obs['pattern']!='no_pattern'].copy()

    adata_test = adata[adata.obs['pattern']==testPattern]
    adata_control = adata[adata.obs['pattern']!=testPattern]

    sample_size = int(len(adata_test)/len(adata_control.obs['pattern'].unique()))
    pattern_groups = adata_control.obs.groupby('pattern')
    subset_obs_list = []

    # loop over group patterns, for each pattern, group by n_spots_interval. In those interval groups, sample `target_samplesize`. 
    for pattern, group in pattern_groups:
        subset = group['n_spots_interval'].sample(n=sample_size, random_state=random_seed)
        subset_obs_list.append(subset)
    
    # Concatenate all subset observations and use these indices to subset the adata_control AnnData object
    all_subset_obs = pd.concat(subset_obs_list)
    adata_control_subset = adata_control[adata_control.obs.index.isin(all_subset_obs.index)]
    
    # Concatenate adata_test and adata_control_subset
    adata = ad.concat([adata_test, adata_control_subset])
    adata.obs[testPattern] = np.where(adata.obs["pattern"] == testPattern, testPattern, "other")

    return adata