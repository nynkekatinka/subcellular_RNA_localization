import pandas as pd
import numpy as np
import sklearn.model_selection as skm


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


def subsetGenes(adata, pattern: str = 'pericellular', pattern_strength: int = 0.9, count_threshold: int = 11, high_or_low: str = 'low'):
    """
    Subset the anndata object into a `1 gene multiple cells` object. Can filter the cells based on the number of spots, the pattern and the pattern strength.

    Parameters
    ----------
    adata : ad.AnnData object
        complete anndata object.
    pattern : str
        Type of subcellular expression pattern you want to filter on. Default is 'pericellular', which has the highest f1 RF score.
    pattern_strength : int
        strength of the pattern, which is labeled as prop in the anndata object. Default is 0.9.
    high_or_low : str
        Whether you want to filter genes with a higher or lower count than the given threshold. Default is lower.
    count_threshold : int
        Count threshold to filter on. Default is 11, so that genes with count 0-10 are selected. For high, the threshold is 140.

    Returns
    -------
    ad.AnnData

    """
    adata_filtered = adata[(adata.obs['pattern'] == pattern) & 
                           (adata.obs['n_spots'] < count_threshold if high_or_low == 'low' else adata.obs['n_spots'] > count_threshold) & 
                           (adata.obs['prop'] == pattern_strength)].copy()

    
    subset_dict = {}
    i = 0
    while len(adata_filtered.obs['cell_id'].unique()) > 0:
        # Group by 'cell_id'
        grouped = adata_filtered.obs.groupby('cell_id')

        # For each group, select at most one item
        subsets = [group.sample(n=1) for _, group in grouped]

        # Concatenate the results back into a DataFrame
        subset_obs = pd.concat(subsets)

        # Use this DataFrame to subset the original anndata object and store in dictionary, while omitting the selected cells from the original adata object
        count_subset = adata_filtered[adata_filtered.obs.index.isin(subset_obs.index)].copy()
        subset_dict[i] = count_subset
        adata_filtered = adata_filtered[~adata_filtered.obs.index.isin(subset_obs.index)].copy()
        i += 1

    return subset_dict
