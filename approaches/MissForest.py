import numpy as np
from tqdm import tqdm
import pandas as pd
from missingpy import MissForest

def reconstruct(dataset, config_idx):
    print('Reconstructing using MissForest...')

    train_data = dataset.orig_ds['train_X']
    mask = dataset.miss_masks[config_idx]['train_X']

    (datasetLen, dim) = np.shape(train_data)
    incomplete_dataset = np.zeros((datasetLen, dim))

    # IterativeImputer requires corrupted entries to be identified as NaN
    # Using the mask to replace in the input dataset all zero entries for NaN
    for i in range(datasetLen):
        frame = train_data.loc[i, :]
        ms = mask.loc[i, :]
        ms.values[ms.values == 0] = np.nan
        incomplete_dataset[i] = frame.values*ms.values

    incomplete_dataset = pd.DataFrame(incomplete_dataset)

    imputer = MissForest(max_iter=1, verbose=1)
    reconstructed_dataset = imputer.fit_transform(incomplete_dataset)

    print(np.shape(reconstructed_dataset))
    print(reconstructed_dataset)

    return pd.DataFrame(reconstructed_dataset)