import numpy as np
import pandas as pd

def reconstruct(dataset, mask):
    (datasetLen, dim) = np.shape(dataset)

    incomplete_dataset = pd.DataFrame(dataset.copy())
    reconstructed_dataset = pd.DataFrame(np.zeros((datasetLen, dim)))

    for i in range(datasetLen):
        frame = incomplete_dataset.loc[i, :]
        most_frequent = frame.mode()[0]
        ms = mask.loc[i, :]
        frame.values[ms.values == 0] = most_frequent
        reconstructed_dataset.loc[i, :] = frame.values

    return reconstructed_dataset


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
