import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import torch
import matplotlib.pyplot as plt


def plotReconstruction(im_dataset, model, out_file: str, nr_images: int = 4, title: str = ""):
    '''
    This piece of code is nasty, i'm so sorry
    '''

    # Set seed the same as i've been doing for the last 2 years cause I'm scared of change
    seed = 101
    torch.manual_seed(seed)
    import random
    random.seed(seed)
    np.random.seed(seed)


    # Make the same train/val split as I do during training cause I wanna see the reconstruction on non-seen data
    train_percentage = 0.8
    train_data, val_data = torch.utils.data.random_split(im_dataset, [int(train_percentage*len(im_dataset.scaled_data)),len(im_dataset.scaled_data) - int(train_percentage*len(im_dataset.scaled_data))])

    # Set figure size for high-res saving
    plt.rcParams['figure.figsize'] = [20, 16]

    random_images = []
    random_image_paths = []
    fig, axs = plt.subplots(2, nr_images)
    for i in range(nr_images):
        # Get random image
        random_nr = random.randint(0, val_data.dataset.__len__())
        # Get image the way the model sees it for reconstruction
        random_image = val_data.dataset[random_nr].unsqueeze(dim=0)

        # Get the channels split so it plots better
        random_gene = val_data.dataset[random_nr].cpu().detach().numpy()[0,:,:]
        random_dapi = val_data.dataset[random_nr].cpu().detach().numpy()[1,:,:]
        random_sum = random_gene + random_dapi

        random_images.append(random_image)
        random_image_paths.append(val_data.dataset.image_list[random_nr])

        axs[0,i].imshow(random_sum)

    # Now that I have 4 random images, we reconstruct them
    results = []
    for random_image in random_images:                                                  
        result, _, _ = model["cvae"](random_image.to(model["cvae_state_dict"]['encoder_conv.0.weight'].device))                                    
        result = result.cpu().detach()                                                  
        results.append(result)     
                                                   
    for i in range(nr_images):
        gene = torch.squeeze(results[i], dim=0).permute(1, 2, 0)[:,:,0]
        dapi = torch.squeeze(results[i], dim=0).permute(1, 2, 0)[:,:,1]
        sm = gene+dapi

        axs[1,i].imshow(gene)
        axs[1,i].axis("off")
    if title:
        axs[0,0].set_title(title)
    else:
        axs[0,0].set_title(f"ld: {model['ld']}, lr: {model['lr']}, beta:{model['beta']}, epochs{model['epochs']}")

    plt.savefig(out_file)

def calcLatent(im_dataset, model) -> np.ndarray:
    '''
    Calculate the latent space of a model/dataset combo
    This is also kind of a mess, it's an old piece of code that I'm too scared to change anything in.
    '''

    # Make a train loader to do the calculation in batches
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


def convertImageDatasetIntoAnnData(im_dataset, model) -> ad.AnnData:
    '''
    This function take a model/dataset combo and does all the parsing that is necessary to get to an AnnData object that holds the latent space and all metadata.
    '''
    latent = calcLatent(im_dataset, model)
    latent_df = pd.DataFrame(latent, columns=["LatentDim{}".format(i+1) for i in range(latent.shape[1])])

    adata = ad.AnnData(X=latent_df)
    adata.obsm["latent"] = latent
    adata.uns["latent_df_keys"] = list(latent_df.columns) 
    
    try:
        patterns = [os.path.basename(img_path).split("_")[0] for img_path in im_dataset.image_list]
        #if the first thing is a tile then it's real spatial data, so then there's no pattern to find in the ground truth
        patterns = [x if 'tile' not in x else 'spatial' for x in patterns]
        rand_or_pat = ["random" if el == "random" else "pattern" for el in patterns ]
        adata.obs["pattern"] = patterns
        adata.obs["random_or_pattern"] = rand_or_pat
    except:
        print("no patterns")

    # Find number of spots
    n_spots = []
    for img_path in im_dataset.image_list:
        try:
            spots = int(re.findall(r'\d+',re.findall(r'spots\d+',os.path.basename(img_path))[0])[0])
        except:
            spots = int(re.findall(r'\d+',re.findall(r'\d+spots',os.path.basename(img_path))[0])[0])
        n_spots.append(spots)
    adata.obs["n_spots"] = n_spots

    # Bin the number of spots so we can plot it later in discrete categories
    bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, float('Inf')]
    labels = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "70-80","80-90","90-100","100+"]
    n_spots_interval = pd.cut(adata.obs['n_spots'], bins=bins, labels=labels)
    adata.obs["n_spots_interval"] = n_spots_interval

    # Find cell identity
    cell_ids = []
    for img_path in im_dataset.image_list:
        try:
            cell_id = re.findall(r'cellID(\d+)',os.path.basename(img_path))[0]
        except IndexError:
            cell_id = "blank"
        cell_ids.append(cell_id)
    adata.obs["cell_id"] = cell_ids
        
    # Find whether it has a gene name
    genes = []
    for img_path in im_dataset.image_list:
        try:
            gene = re.findall(r'genName([a-zA-Z0-9]+)',os.path.basename(img_path))[0]
        except IndexError:
            gene = "blank"
        genes.append(gene)
    adata.obs["genes"] = genes

    # Find rotation (only for simulated ata)
    try:
        rotation = [int(re.findall('rotated\d+', os.path.basename(img_path))[0].strip('rotated')) for img_path in im_dataset.image_list]
        adata.obs["rotation"] = rotation
        adata.obs["rotation_interval"] =  pd.cut(adata.obs['rotation'], bins=[0, 60, 120, 180, 240, 300, float('Inf')], labels=["0-60", "60-120", "120-180", "180-240", "240-300", "300+"])
    except (IndexError,KeyError):
        print("no rotation")

    # Find correspoinding dapis
    try:
        corresponding_dapis = im_dataset.corresponding_dapis
        adata.obs["corresponding_dapis"] = corresponding_dapis
    except:
        print("no corresponding dapis")
        
    # Get paths to input data as well
    adata.obs["original_image_paths"] = im_dataset.image_list
    
    return adata
