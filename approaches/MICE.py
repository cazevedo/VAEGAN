import numpy as np
from tqdm import tqdm
import pandas as pd
from fancyimpute import IterativeImputer

def reconstruct(dataset, mask):
    incomplete_dataset = np.zeros(np.shape(dataset))

    # IterativeImputer requires corrupted entries to be identified as NaN
    # Using the mask to replace in the input dataset all zero entries for NaN
    for i in range(len(dataset)):
        frame = dataset.loc[i, :]
        ms = mask.loc[i, :]
        ms.values[ms.values == 0] = np.nan

        incomplete_dataset[i] = frame.values*ms.values

    incomplete_dataset = pd.DataFrame(incomplete_dataset)
    print(np.shape(incomplete_dataset))

    n_imputations = 5
    reconstructed_dataset = []
    # IterativeImputer replicates MICE algorithm when used for multiple imputations
    # by applying it repeatedly to the same dataset
    for i in tqdm(range(n_imputations)):
        imputer = IterativeImputer(n_iter=1, sample_posterior=True, random_state=i)
        reconstructed_dataset.append(imputer.fit_transform(incomplete_dataset))

    reconstructed_dataset_mean = np.mean(reconstructed_dataset, axis=0)
    reconstructed_dataset_std = np.std(reconstructed_dataset, axis=0)

    return pd.DataFrame(reconstructed_dataset_mean)

## DEBUG TOOLS ##
# import reconstruct as rc
# import matplotlib.pyplot as plt
# if __name__ == "__main__":
#     original_dataset, incomplete_dataset, mask = rc.get_dataset(mode='MCAR', n_samples=100)
#
#     original_dataset = pd.DataFrame(original_dataset)
#     incomplete_dataset = pd.DataFrame(incomplete_dataset)
#     mask = pd.DataFrame(mask)
#
#     reconstructed_dataset = reconstruct(incomplete_dataset, mask)
#
#     inc = incomplete_dataset.loc[0,:]
#     rec = reconstructed_dataset.loc[0,:]
#     orig = original_dataset.loc[0,:]
#
#     print(np.shape(inc))
#     print(np.shape(rec))
#     print(np.shape(orig))
#
#     samples = np.vstack([inc, rec, orig])
#     fig = rc.plot(samples)
#     plt.savefig('Multiple_Impute_out1/{}.png'.format(str(0).zfill(3)), bbox_inches='tight')
#     plt.close(fig)



# from sklearn.linear_model import LinearRegression
# import os
# import sys
# projectdir = os.path.dirname(__file__)
# app_path = os.path.join(projectdir, 'scikit-mice')
# sys.path.insert(0, app_path)
# import skmice
#
# from statsmodels.imputation import mice
# import statsmodels.api as sm

# np.set_printoptions(linewidth=115, suppress=False, precision=1, floatmode='fixed')
#
# def gendat():
#     """
#     Create a data set with missing values.
#     """
#
#     np.random.seed(34243)
#
#     n = 20
#     p = 5
#
#     exog = np.random.normal(size=(n, p))
#     exog[:, 0] = exog[:, 1] - exog[:, 2] + 2*exog[:, 4]
#     exog[:, 0] += np.random.normal(size=n)
#     exog[:, 2] = 1*(exog[:, 2] > 0)
#
#     endog = exog.sum(1) + np.random.normal(size=n)
#
#     df = pd.DataFrame(exog)
#     df.columns = ["x%d" % k for k in range(1, p+1)]
#
#     df["y"] = endog
#
#     # df.x1[0:60] = np.nan
#     # df.x2[0:40] = np.nan
#     df.x1[0:5] = np.nan
#     df.x2[15:19] = np.nan
#     df.x3[10:30:2] = np.nan
#     df.x4[20:50:3] = np.nan
#     df.x5[40:45] = np.nan
#     df.y[30:100:2] = np.nan
#
#     return df
#
# def reconstruct2(dataset, mask):
#     incomplete_dataset = np.zeros(np.shape(dataset))
#
#     # IterativeImputer requires corrupted entries to be identified as NaN
#     # Using the mask to replace in the input dataset all zero entries for NaN
#     for i in range(len(dataset)):
#         frame = dataset.loc[i, :]
#         ms = mask.loc[i, :]
#         ms.values[ms.values == 0] = np.nan
#
#         incomplete_dataset[i] = frame.values*ms.values
#
#     incomplete_dataset = pd.DataFrame(incomplete_dataset)
#     incomplete_dataset.columns = map(str, incomplete_dataset.columns.values)
#
#     incomplete_dataset.columns = [item + ':' for item in incomplete_dataset.columns]
#
#     print(incomplete_dataset.columns)
#
#     # sys.exit(0)
#
#     # print(incomplete_dataset)
#
#     reconstructed_dataset = mice.MICEData(incomplete_dataset)
#     # print(np.shape(imp_data))
#     print(np.shape(reconstructed_dataset.data))
#     print(reconstructed_dataset.data)
#
#     # mi = mice.MICE("y ~ x1 + x2 + x1:x2", sm.OLS, reconstructed_dataset)
#     mi = mice.MICE("0", sm.OLS, reconstructed_dataset)
#     results = mi.fit(n_burnin=10, n_imputations=10)
#
#     print(np.shape(reconstructed_dataset.data))
#
#     sys.exit(0)
#
#     return pd.DataFrame(reconstructed_dataset)
#

# if __name__ == "__main__":
#     original_dataset, dataset, mask = rc.get_dataset(mode='MCAR', n_samples=100)
#
#     original_dataset = pd.DataFrame(original_dataset)
#     dataset = pd.DataFrame(dataset)
#     mask = pd.DataFrame(mask)
#
#     incomplete_dataset = np.zeros(np.shape(dataset))
#
#     for i in range(len(dataset)):
#         frame = dataset.loc[i, :]
#         ms = mask.loc[i, :]
#         ms.values[ms.values == 0] = np.nan
#
#         incomplete_dataset[i] = frame.values*ms.values
#
#     print(np.shape(incomplete_dataset))
#     # print(incomplete_dataset[0:1,:].reshape((2, -1)))
#
#     imputer = IterativeImputer(missing_values=np.nan, n_iter=2, sample_posterior=True, random_state=1)
#     # reconstructed_dataset = imputer.fit_transform(incomplete_dataset[0, :].reshape((2, -1)))
#     reconstructed_dataset = imputer.fit_transform(incomplete_dataset)
#     # reconstructed_dataset = imputer.complete(incomplete_dataset)
#
#
#     # print(reconstructed_dataset_mean.shape)
#     print(np.shape(reconstructed_dataset))
#     # print(reconstructed_dataset)
#
#     # fig = rc.plot([reconstructed_dataset])
#     # plt.savefig('Multiple_Impute_out1/{}.png'.format(str(0).zfill(3)), bbox_inches='tight')
#     # plt.close(fig)
#
#     # sys.exit(0)