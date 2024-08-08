# subcellular_RNA_localization

Code used in the master thesis of Nynke Tilkema.
Environment can be found in environment.yml
# Blurry: detecting subcellular expression patterns using blurred single-cell images


## Order of operations

- load trained torch model `io_data.loadBlurry`
- load merfish dataset `io_data.loadSpatialDataset`
- Convert merfish dataset to anndata while parsing metadata `utils.convertImageDatasetIntoAnnData`
- Calculate umap `analyze.calcUmap`
