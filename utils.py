import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch

def calcLatent(im_dataset, model):
    batch_size = 512
    train_loader = torch.utils.data.DataLoader(im_dataset, batch_size = batch_size, shuffle=False)
    with torch.no_grad():
        model["cvae"].eval()
        model["cvae"].to(model["device"])
        latent_list = []
        for batch in tqdm(train_loader):
            latent, _, _ = model["cvae"].embed(batch.to(model["device"]))
            latent_list.append(latent.to("cpu"))
        latent = torch.concat(latent_list, dim=0)
        latent = latent.numpy()
    return latent


def convertImageDatasetIntoAnnData(im_dataset, model):
    latent = calcLatent(im_dataset, model)

    latent_df = pd.DataFrame(latent, columns=["LatentDim{}".format(i+1) for i in range(latent.shape[1])])

    combined_data = ad.AnnData(X=latent_df)
    combined_data.obsm["latent"] = latent
    combined_data.uns["latent_df_keys"] = list(latent_df.columns) 
    
    try:
        patterns = [os.path.basename(img_path).split("_")[0] for img_path in im_dataset.image_list]
        patterns = [x if 'tile' not in x else 'spatial' for x in patterns]
        rand_or_pat = ["random" if el == "random" else "pattern" for el in patterns ]
        combined_data.obs["pattern"] = patterns
        combined_data.obs["random_or_pattern"] = rand_or_pat
    except:
        print("no patterns")
    # Real data
    n_spots = []
    for img_path in im_dataset.image_list:
        try:
            spots = int(re.findall(r'\d+',re.findall(r'spots\d+',os.path.basename(img_path))[0])[0])
        except:
            spots = int(re.findall(r'\d+',re.findall(r'\d+spots',os.path.basename(img_path))[0])[0])
        n_spots.append(spots)
    combined_data.obs["n_spots"] = n_spots
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('Inf')]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "70-80","80-90","90-100","100+"]
    n_spots_interval = pd.cut(combined_data.obs['n_spots'], bins=bins, labels=labels)
    combined_data.obs["n_spots_interval"] = n_spots_interval

    cell_ids = []
    for img_path in im_dataset.image_list:
        try:
            cell_id = re.findall(r'cellID(\d+)',os.path.basename(img_path))[0]
        except IndexError:
            cell_id = "blank"
        cell_ids.append(cell_id)
    
    combined_data.obs["cell_id"] = cell_ids
        
        
    genes = []
    for img_path in im_dataset.image_list:
        try:
            gene = re.findall(r'genName([a-zA-Z0-9]+)',os.path.basename(img_path))[0]
        except IndexError:
            gene = "blank"
        genes.append(gene)
    combined_data.obs["genes"] = genes

    try:
        rotation = [int(re.findall('rotated\d+', os.path.basename(img_path))[0].strip('rotated')) for img_path in im_dataset.image_list]
        combined_data.obs["rotation"] = rotation
        combined_data.obs["rotation_interval"] =  pd.cut(combined_data.obs['rotation'], bins=[0, 60, 120, 180, 240, 300, float('Inf')], labels=["0-60", "60-120", "120-180", "180-240", "240-300", "300+"])
    except (IndexError,KeyError):
        print("no rotation")

    try:
        corresponding_dapis = im_dataset.corresponding_dapis
        combined_data.obs["corresponding_dapis"] = corresponding_dapis
    except:
        print("no corresponding dapis")
        
    combined_data.obs["original_image_paths"] = im_dataset.image_list
    
    return combined_data
