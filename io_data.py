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


def loadBlurry(path_to_model: Union[Path, str] = "/media/gambino/students_workdir/nynke/blurry_model/blurry_torch_model", device=0):
    '''
    Load trained blurry torch model
    '''
    blurry = torch.load(path_to_model, map_location=torch.device(0))
    blurry["device"] = device
    return blurry

def loadBlurryAdata(path_to_adata: Union[Path, str] = "/media/gambino/students_workdir/nynke/blurry_model/simulated_trained_adata.h5ad"):
    '''
    Load anndata object that contains latent space of training data
    '''
    adata = sc.read_h5ad(path_to_adata)
    return adata
    

def loadSpatialDataset(single_gene_images_glob: str, model: str = None, adata_to_concat: ad.AnnData = None) -> ad.AnnData:
    """loadSpatialDataset.

    Parameters
    ----------
    single_gene_images_glob : str
        glob pattern pointing to all single gene, single cell images that need to be loaded in the dataset
    model : str
        path to torch model that should be used to create latent representations, default points to where the model is on my machine, so do not use.
    adata_to_concat : ad.AnnData
        anndata object to concatenate the loaded single images to. This allows you to recursively call this function, but als is mostly meant so you can load trained anndata, and concatenate it to the real spatial data you create with this function.
        Default is not to concat anything.

    Returns
    -------
    ad.AnnData

    """

    if model is None:
        model = loadBlurry()
    else:
        model = loadBlurry(model)

    im_dataset = ImageDatasetWithDapis(single_gene_images_glob)
    print(type(im_dataset))
    adata = convertImageDatasetIntoAnnData(im_dataset, model)

    if adata_to_concat is not None:
        if adata_to_concat == "blurry":
            adata_to_concat = loadBlurryAdata()
        else:
            adata_to_concat = loadBlurryAdata(adata_to_concat)

        adata = adata_to_concat.concatenate(adata)
    return adata


class ImageDatasetWithDapis(torch.utils.data.Dataset):
    def __init__(self, glob_pattern: Union[List, str], dapi_dir: str = None):
        """__init__.

        Parameters
        ----------
        glob_pattern : Union[List, str]
            glob pattern pointing to the single-gene images that need to be loaded, or a list of the paths.
        dapi_dir : str
            path to the directory where the corresponding dapis are located. Default is none, and assumes that they are located in a directory called 'dapi_zoomed' on the the parent level of the glob_pattern. It is also assumed all your dapi images follow a pattern of {single_gene_path.stem}_DAPI.tif 
        """

        convert_tensor = transforms.ToTensor()
        
        if isinstance(glob_pattern, str):
            self.image_list = sorted(glob(glob_pattern))
        else:
            self.image_list = glob_pattern

        # First compile list of all the corresponding dapis to the gene images.
        self.corresponding_dapis = []
        for image_path in self.image_list:
            image_path = Path(image_path)
            if dapi_dir is None:
                corresponding_dapi_file = image_path.parent / "dapi_zoomed" / f"{image_path.stem}_DAPI.tif"
            else:
                corresponding_dapi_file = Path(dapi_dir) / f"{image_path.stem}_DAPI.tif"
            self.corresponding_dapis.append(corresponding_dapi_file)

        scaled_images = []
        scaled_dapis = []

        # now we prepare the image-dapi pairs for passes through the model
        for image, dapi in tqdm(zip(self.image_list, self.corresponding_dapis), desc="Loading images", unit="image", total = len(self.image_list)):
            if ".tif" not in image:
                continue

            im = io.imread(image)
            dapi = io.imread(dapi)
            dapi = dapi.astype(np.float32)
            image_tensor = convert_tensor(im).float()
            dapi_tensor =  convert_tensor(dapi).float()
            # Every once in a while something goes wrong and an empty image is passes, an normalizing then divides by zero
            try:
                image_tensor = transforms.Normalize(torch.mean(image_tensor),torch.std(image_tensor))(image_tensor)
                dapi_tensor = transforms.Normalize(torch.mean(dapi_tensor),torch.std(dapi_tensor))(dapi_tensor)
            except ValueError:
                print(image, dapi)
                continue

            # Create an extra first dimension which is used for the batches in training: result shape = [1,100,100]
            image_tensor = torch.unsqueeze(image_tensor, dim=0)
            dapi_tensor = torch.unsqueeze(dapi_tensor, dim=0)
        
            scaled_image = self.scale(image_tensor, 0, 1)
            scaled_images.append(scaled_image)
            scaled_dapi = self.scale(dapi_tensor, 0, 1)
            scaled_dapis.append(scaled_dapi)

        #convert list of images to torch array of images
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
    from analyze import calcUmap, plotUmap
    adata = loadSpatialDataset(single_gene_images_glob = "/media/gambino/students_workdir/nynke/blurry_model/ileum_data/mouse_ileum/coordinate_parsed/*.tif", adata_to_concat="blurry")
    calcUmap(adata)
    plotUmap(adata, "pattern", "./test_real_data_overlay.png", pattern_to_alpha = "spatial")

