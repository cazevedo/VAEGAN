import numpy as np
import pandas as pd
from tqdm import tqdm

# Mean imputation method that uses the mean of each row as imputation of the missing values
# (suitable for images datasets)
def reconstruct(dataset, mask):
    print('Reconstructing using Mean Imputation...')

    (datasetLen, dim) = np.shape(dataset)

    incomplete_dataset = dataset.copy()
    reconstructed_dataset = pd.DataFrame(np.zeros((datasetLen, dim)))

    for i in tqdm(range(dim)):
        # print(incomplete_dataset)
        frame = incomplete_dataset.iloc[:, i]
        most_frequent = frame.mean()
        ms = mask.iloc[:, i]

        frame.values[ms.index[ms == 0]] = most_frequent
        reconstructed_dataset.iloc[:, i] = frame.values

    return reconstructed_dataset

# def reconstruct_tabular(dataset, mask):
#     print('Reconstructing using Most Frequent...')
#     (datasetLen, dim) = np.shape(dataset)
#
#     incomplete_dataset = pd.DataFrame(dataset.copy())
#     reconstructed_dataset = pd.DataFrame(np.zeros((datasetLen, dim)))
#
#     for i in tqdm(range(dim)):
#         frame = incomplete_dataset.loc[:, i]
#         mean = frame.mean()
#         ms = mask.loc[:, i]
#         frame.values[ms.index[ms == 0]] = mean
#         reconstructed_dataset.loc[:, i] = frame.values
#
#     return reconstructed_dataset

## DEBUG TOOLS ##
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import reconstruct as rc
# if __name__ == "__main__":
#     original_dataset, incomplete_dataset, mask = rc.get_dataset(mode='MNAR', n_samples=20)
#
#     original_dataset = pd.DataFrame(original_dataset)
#     incomplete_dataset = pd.DataFrame(incomplete_dataset)
#     mask = pd.DataFrame(mask)
#
#     reconstructed_dataset = reconstruct2(incomplete_dataset, mask)
#
#     inc = incomplete_dataset.loc[0, :]
#     rec = reconstructed_dataset.loc[0,:]
#     orig = original_dataset.loc[0,:]
#
#     samples = np.vstack([inc, rec, orig])
#     fig = rc.plot(samples)
#     plt.savefig('Multiple_Impute_out1/{}.png'.format(str(0).zfill(3)), bbox_inches='tight')
#     plt.close(fig)
