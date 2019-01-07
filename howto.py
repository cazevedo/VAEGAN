# Config
approach = "mean"
dataset_name = "MNIST"
miss_strats = ['MCAR'] # ['MAR','MAR','MCAR']
one_strat = miss_strats[0]
missing_ratio = 0.5
N = 2
train_ratio = 0.9

# Getting a dataset
from get_datasets import dataset_folder
original=dataset_folder(
	dataset=dataset_name,
	miss_strats=miss_strats,
	miss_rates=missing_ratio,
	n=N,
	train_ratio=train_ratio
)

# Fix the indices from splitting train/test
original.orig_ds['train_X'].reset_index(inplace=True)
for i in range(N):
	original.miss_masks[i]['train_X'].reset_index(inplace=True)

# Select a config index
config_idx = 0 # up to n-1 from above

# For testing purposes
test_len = 20
original.orig_ds['train_X'] = original.orig_ds['train_X'].loc[:test_len, :]
original.miss_masks[config_idx]['train_X'] = original.miss_masks[config_idx]['train_X'].loc[:test_len, :]

if approach == "mean":
	from approaches import MeanImputation
	reconstructed = MeanImputation.reconstruct(original, config_idx)
else:
	raise NotImplementedError(approach)

# Save the reconstructed and original
import pickle
fn_fmt = "{approach}_{dataset}_{missing_ratio}_{missing_mechanism}_{n}.pkl"
fn = fn_fmt.format(
	approach=approach,
	dataset=dataset_name,
	missing_ratio=missing_ratio,
	missing_mechanism=one_strat,
	n=config_idx
)
with open("results/"+fn, "wb") as f:
	pickle.dump((original, reconstructed), f)