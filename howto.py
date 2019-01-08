# Config
approach = "mean"
dataset_name = "MNIST"
miss_strats = ['MCAR'] # ['MAR','MAR','MCAR']
mechanism = miss_strats[0]
missing_ratio = 0.5
N = 2
train_ratio = 0.9

# Getting a dataset
from get_datasets import dataset_folder
dataset=dataset_folder(
	dataset=dataset_name,
	miss_strats=miss_strats,
	miss_rates=missing_ratio,
	n=N,
	train_ratio=train_ratio
)

# Fix the indices from splitting train/test
dataset.orig_ds['train_X'].reset_index(inplace=True)
for i in range(N):
	dataset.miss_masks[i]['train_X'].reset_index(inplace=True)

# Select a config index
config_idx = 0 # up to n-1 from above

# For testing purposes
# test_len = 20
# dataset.orig_ds['train_X'] = dataset.orig_ds['train_X'].loc[:test_len, :]
# dataset.miss_masks[config_idx]['train_X'] = dataset.miss_masks[config_idx]['train_X'].loc[:test_len, :]

if approach == "mean":
	from approaches import MeanImputation
	reconstructed = MeanImputation.reconstruct(dataset, config_idx)
elif approach == "frequent":
	from approaches import MostFrequent
	reconstructed = MostFrequent.reconstruct(dataset, config_idx)
elif approach == "MICE":
	from approaches import MICE
	reconstructed = MICE.reconstruct(dataset, config_idx)
else:
	raise NotImplementedError(approach)

# Save the original dataset and reconstructed
import pickle

fn_org = "{dataset}_{train_ratio}.pkl".format(
	dataset=dataset_name,
	train_ratio=train_ratio
)
with open("results/"+fn_org, "wb") as f:
	pickle.dump((dataset.orig_ds['train_X'], dataset.orig_ds['test_X'], dataset_name, train_ratio), f)

fn_fmt = "{dataset}_{approach}_{missing_mechanism}_{missing_ratio}_{n}.pkl"
fn = fn_fmt.format(
	dataset=dataset_name,
	approach=approach,
	missing_mechanism=mechanism,
	missing_ratio=missing_ratio,
	n=config_idx
)

with open("results/"+fn, "wb") as f:
	pickle.dump((reconstructed, dataset_name, approach, mechanism, missing_ratio, config_idx), f)
