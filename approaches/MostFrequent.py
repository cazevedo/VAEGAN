import numpy as np
import pandas as pd
from tqdm import tqdm

# Most frequent imputation method that uses the most frequent element of each row
# as imputation of the missing values (suitable for images datasets)
def reconstruct(dataset, mask):
    print('Reconstructing using Most Frequent...')
    (datasetLen, dim) = np.shape(dataset)

    incomplete_dataset = pd.DataFrame(dataset.copy())
    reconstructed_dataset = pd.DataFrame(np.zeros((datasetLen, dim)))

    for i in tqdm(range(datasetLen)):
        frame = incomplete_dataset.loc[i, :]
        most_frequent = frame.mode()[0]
        ms = mask.loc[i, :]

        frame.values[ms.values == 0] = most_frequent
        reconstructed_dataset.loc[i, :] = frame.values

    return reconstructed_dataset

# Most frequent imputation method that uses the most frequent frequent of each column
# as imputation of the missing values (suitable for tabular datasets)
def reconstruct_tabular(dataset, mask):
    print('Reconstructing using Most Frequent...')
    (datasetLen, dim) = np.shape(dataset)

    incomplete_dataset = pd.DataFrame(dataset.copy())
    reconstructed_dataset = pd.DataFrame(np.zeros((datasetLen, dim)))

    for i in tqdm(range(dim)):
        frame = incomplete_dataset.loc[:, i]
        most_frequent = frame.mode()[0]
        ms = mask.loc[:, i]

        frame.values[ms.index[ms == 0]] = most_frequent
        reconstructed_dataset.loc[:, i] = frame.values

    return reconstructed_dataset


## DEBUG TOOLS ##
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import reconstruct as rc
# if __name__ == "__main__":
#     original_dataset, incomplete_dataset, mask = rc.get_dataset(mode='MCAR', n_samples=20)
#
#     reconstructed_dataset = reconstruct(incomplete_dataset, mask)
#
#     inc = incomplete_dataset.loc[3, :]
#     rec = reconstructed_dataset.loc[3,:]
#     orig = original_dataset.loc[3,:]
#
#     samples = np.vstack([inc, rec, orig])
#     fig = rc.plot(samples)
#     plt.savefig('Multiple_Impute_out1/{}.png'.format(str(0).zfill(3)), bbox_inches='tight')
#     plt.close(fig)
