import os
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
import scanpy as sc
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from skimage import io
from io_data import loadSpatialDataset

seed = 101
import random
random.seed(seed)
np.random.seed(seed)

###################################
##Make palette for umap plotting ##
###################################
tmp_patterns = ["spatial","random", "intranuclear", "nuclear-edge", "extranuclear", "foci", "protrusion", "cell-edge", "perinuclear", "pericellular","pattern"]

def get_cmap(n, name='tab10'):
    ##Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    ##RGB color; the keyword argument name must be a standard mpl colormap name.
    return plt.cm.get_cmap(name, n)

global_cmap = get_cmap(len(tmp_patterns))
palette = {}
for i, el in enumerate(tmp_patterns):
    palette[el] =  global_cmap(i)

####################################

def calcUmap(adata, nr_latent_dims = 15, n_neighbors = 25):
    sc.tl.pca(adata, svd_solver='arpack', n_comps=nr_latent_dims - 1)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=nr_latent_dims)
    sc.tl.umap(adata)

def plotUmap(adata, color, out_file, title="", pattern_to_alpha = None, s=16, legend_fontsize=20):
    if pattern_to_alpha is not None:
        alpha_list = [1 if pattern == pattern_to_alpha else 0.2 for pattern in adata.obs["pattern"]]
    else:
        alpha_list = None
    fig = sc.pl.umap(adata, color=color, title=title, s=s, palette=palette, alpha=alpha_list, legend_fontsize = legend_fontsize, return_fig = True)
    fig.savefig(out_file)

if __name__ == '__main__':
    ## Full workflow for creating new dataset plot
    adata = loadSpatialDataset(single_gene_images_glob = "/mnt/data/MERFISH/colonCancer1/parsed/coordinate_parsed/tile49/*.tif")
    calcUmap(adata)
    plotUmap(adata, "n_spots", "./test_umap_plot.png", title="test")
