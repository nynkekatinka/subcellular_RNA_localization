import re
import os
import time
import torch
import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from glob import glob
from tqdm import tqdm
from PIL import Image
from skimage import io
from pathlib import Path
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Union, List
from utils import convertImageDatasetIntoAnnData
from torchvision import datasets, transforms
Image.MAX_IMAGE_PIXELS = None 


def loadBlurry(path_to_model: Union[Path, str] = "/mnt/data/david/SEP/models/supervised_with_dapi/2023-05-05_cvae_even_less_spots_dual_2classifiers_100weight_coordinateparsed_channel_dapi_with_labels_correctrotation_zoomed_fixedlocation_unbalanced_lr1e-05_classlr1e-03_nclasses6_beta1e-04_15ld_100epochs"):
    blurry = torch.load(path_to_model)
    return blurry

def loadBlurryAdata(path_to_adata: Union[Path, str] = "/mnt/data/david/SEP/embeddings/2023-05-05_cvae_even_less_spots_dual_2classifiers_100weight_coordinateparsed_channel_dapi_with_labels_correctrotation_zoomed_fixedlocation_unbalanced_lr1e-05_classlr1e-03_nclasses6_beta1e-04_15ld_100epochs_adata.h5ad"):
    adata = sc.read_h5ad(path_to_adata)
    return adata
    

def loadSpatialDataset(single_gene_images_glob: str, model=None, adata_to_concat: ad.AnnData = None):
    if model is None:
        model = loadBlurry()
    else:
        model = loadBlurry(model)

    im_dataset = ImageDatasetWithDapis(single_gene_images_glob)
    adata = convertImageDatasetIntoAnnData(im_dataset, model)

    if adata_to_concat is not None:
        if adata_to_concat == "blurry":
            adata_to_concat = loadBlurryAdata()
        else:
            adata_to_concat = loadBlurryAdata(adata_to_concat)

        adata = adata_to_concat.concatenate(adata)
    return adata


class ImageDatasetWithDapis(torch.utils.data.Dataset):
    def __init__(self, glob_pattern):
        convert_tensor = transforms.ToTensor()
        
        self.image_list = sorted(glob(glob_pattern))

        self.corresponding_dapis = []
        for image_path in self.image_list:
            image_path = Path(image_path)
            corresponding_dapi_file = image_path.parent / "dapi_zoomed" / f"{image_path.stem}_DAPI.tif"
            self.corresponding_dapis.append(corresponding_dapi_file)

        scaled_images = []
        scaled_dapis = []
        for image, dapi in tqdm(zip(self.image_list, self.corresponding_dapis), desc="Loading images", unit="image"):
            if ".tif" not in image:
                continue

            # im = Image.open(image)
            im = io.imread(image)
            dapi = io.imread(dapi)
            dapi = dapi.astype(np.float32)
            image_tensor = convert_tensor(im).float()
            dapi_tensor =  convert_tensor(dapi).float()
            try:
                image_tensor = transforms.Normalize(torch.mean(image_tensor),torch.std(image_tensor))(image_tensor)
                dapi_tensor = transforms.Normalize(torch.mean(dapi_tensor),torch.std(dapi_tensor))(dapi_tensor)
            except ValueError:
                print(image, dapi)
                continue

            image_tensor = torch.unsqueeze(image_tensor, dim=0)
            dapi_tensor = torch.unsqueeze(dapi_tensor, dim=0)
        

            scaled_image = self.scale(image_tensor, 0, 1)
            scaled_images.append(scaled_image)
            scaled_dapi = self.scale(dapi_tensor, 0, 1)
            scaled_dapis.append(scaled_dapi)

        self.scaled_data = torch.cat(scaled_images, dim=0)
        self.scaled_dapi = torch.cat(scaled_dapis, dim=0)

    def __getitem__(self, index):
        x = self.scaled_data[index]
        y = self.scaled_dapi[index]
        z = torch.cat((x,y), dim=0)
        return z

    def __len__(self):
        return len(self.scaled_data)
    
    def __copy__(self):
        obj = type(self).__new__(self.__class__)
        obj.__dict__.update(self.__dict__)
        return obj

    def __add__(self, other):
        return_dataset = self.__copy__()
        return_dataset.scaled_data = torch.cat((self.scaled_data, other.scaled_data))
        return_dataset.scaled_dapi = torch.cat((self.scaled_dapi, other.scaled_dapi))
        return_dataset.image_list = self.image_list + other.image_list
        return_dataset.corresponding_dapis = self.corresponding_dapis + other.corresponding_dapis
        return return_dataset

    def __radd__(self, other): 
        return_dataset = self.__copy__()
        return_dataset.scaled_data = torch.cat((self.scaled_data, other.scaled_data))
        return_dataset.scaled_dapi = torch.cat((self.scaled_dapi, other.scaled_dapi))
        return_dataset.image_list = self.image_list + other.image_list
        return_dataset.corresponding_dapis = self.corresponding_dapis + other.corresponding_dapis
        return return_dataset
    
    def subsample(self, n):
        import random
        return_dataset = self.__copy__()
        image_list, scaled_data, scaled_dapi = zip(*random.sample(list(zip(self.image_list, self.scaled_data, scaled_dapi)), n))
        return_dataset.scaled_data = scaled_data
        return_dataset.image_list = image_list
        return_dataset.scaled_dapi = scaled_dapi
        return return_dataset

    def scale(self, tensor, min_value, max_value):
        v_min, v_max = tensor.min(), tensor.max()
        new_min, new_max = min_value, max_value
        v_p = (tensor - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
        return v_p

if __name__ == '__main__':
    # adata = loadSpatialDataset(single_gene_images_glob = "/mnt/data/MERFISH/colonCancer1/parsed/coordinate_parsed/tile49/*.tif", dapi_images_glob = "/mnt/data/MERFISH/colonCancer1/parsed/coordinate_parsed/tile49/dapi_zoomed/*.tif", dapi_prefixes = ["cellID"], adata_to_concat = "/mnt/data/david/SEP/embeddings/2023-05-05_cvae_even_less_spots_dual_2classifiers_100weight_coordinateparsed_channel_dapi_with_labels_correctrotation_zoomed_fixedlocation_unbalanced_lr1e-05_classlr1e-03_nclasses6_beta1e-04_15ld_100epochs_adata.h5ad", model = "/mnt/data/david/SEP/models/supervised_with_dapi/2023-05-05_cvae_even_less_spots_dual_2classifiers_100weight_coordinateparsed_channel_dapi_with_labels_correctrotation_zoomed_fixedlocation_unbalanced_lr1e-05_classlr1e-03_nclasses6_beta1e-04_15ld_100epochs")
    from analyze import calcUmap, plotUmap
    adata = loadSpatialDataset(single_gene_images_glob = "/mnt/data/MERFISH/colonCancer1/parsed/coordinate_parsed/tile49/*.tif", adata_to_concat="blurry")
    calcUmap(adata)
    plotUmap(adata, "pattern", "./test_real_data_overlay.png", pattern_to_alpha = "spatial")


