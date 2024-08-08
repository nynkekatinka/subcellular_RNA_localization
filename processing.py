import os
import anndata as ad
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
import scanpy as sc
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from skimage import io
from io_data import loadSpatialDataset, loadBlurry, loadBlurryAdata
from typing import Tuple


def classify_embedding(model: nn.Module, embedding: torch.Tensor, binary: bool = False) -> Tuple[int, torch.Tensor]:
    """ Classify a singular embedding using a trained model

    Parameters
    ----------
    model : nn.Module
        model to use to classify
    embedding : torch.Tensor
        embedding
    binary : bool
        Flag to decide whether to classify 'pattern / no pattern' or 'which pattern'. Default = False.

    Returns
    -------
    Tuple[int, torch.Tensor]
    Prediction index = int: index of pointing to which of the model.unique_patterns is the classified patterns
    Prediction tensor = torch.Tensor: tensor giving classification scores for all classes


    """

    if binary:
        prediction = model.binary_classifier.forward(embedding)
    else:
        prediction = model.classifier.forward(embedding)
    pred_index = torch.argmax(prediction)
    pred_index = int(pred_index.cpu())
    return pred_index, prediction

def classifyAdata(adata: ad.AnnData, model: nn.Module, device: torch.device):
    """ Take an anndata object with a latent space representation and classify (both binary and more) all data points it contains

    Parameters
    ----------
    adata : ad.AnnData
    model : nn.Module
        Trained model
    device : torch.device
        device to perform the classification on.
    """
    binary_classification_column = []
    pattern_classification_column = []
    pattern_scores = []
    binary_pattern_scores = []
    tmp_device = torch.device("cuda", device) if torch.cuda.is_available() else 'cpu'
    model["cvae"].to(tmp_device)
    
    for i in tqdm(range(len(adata))):

        tens = torch.from_numpy(adata[i].X)
        tens = tens.to(tmp_device)
        
        # class pattern
        pattern_classification_index, prediction = classify_embedding(model["cvae"], tens, binary=False)
        pattern_scores.append(np.array(prediction.detach().to("cpu")))
        
        # class binary
        binary_classification_index, binary_predictions = classify_embedding(model["cvae"], tens, binary=True)
        binary_pattern_scores.append(np.array(binary_predictions.detach().to("cpu")))
        
        binary_classification_column.append(binary_classification_index)
        pattern_classification_column.append(pattern_classification_index)
        

    binary_classification_column = ["random" if el == 0 else "pattern" for el in binary_classification_column]
    adata.obs["binary_classified"] = binary_classification_column
    
    adata.obs["classified"] = pattern_classification_column
    unique_patterns = model["unique_patterns"]
    adata.obs["pattern_classified"] = adata.obs.apply(lambda r: unique_patterns[r["classified"]], axis=1)
        
    pattern_scores_df = pd.DataFrame(np.concatenate(pattern_scores, axis=0), columns=unique_patterns)
    binary_scores_df = pd.DataFrame(np.concatenate(binary_pattern_scores, axis=0), columns=["random_score", "pattern_score"]) 

    # Fix indexed before concatenating so that it works correctly
    pattern_scores_df.index = adata.obs.index
    binary_scores_df.index = adata.obs.index
    adata.obs = pd.concat([adata.obs, pattern_scores_df, binary_scores_df], axis=1)
    return adata

if __name__ == '__main__':
    from analyze import plotUmap
    adata = loadBlurryAdata()
    model = loadBlurry()
    adata = classifyAdata(adata, model, device=0)
    plotUmap(adata, "pattern", "./test_basic_model_umap_pattern_classified.png", pattern_to_alpha = "random")


