import json
import importlib.util
import numpy as np
from tqdm import tqdm
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.append('.')
import get_datasets
import GAINCredit_alpha as gain

### --------------------- DEBUG TOOLS -----------------------###
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys

def plot(samples):
    fig = plt.figure(figsize=(5, 5))
    gs = gridspec.GridSpec(5, 5)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')
        # plt.imshow(sample, cmap='Greys_r')

    return fig

class ReconstructDataset(object):
    def __init__(self):
        # Load json config file (should be in the root dir of the project)
        script_dir = os.path.abspath(__file__)
        (self.filedir, tail) = os.path.split(script_dir)
        (projectdir, tail) = os.path.split(self.filedir)
        # filename = 'config.json'
        filename = 'config2.json'
        abs_file_path = os.path.join(projectdir, filename)

        with open(abs_file_path) as f:
            self.config = json.load(f)

        script_dir = os.path.abspath(__file__)
        (filedir, tail) = os.path.split(script_dir)
        (projectdir, tail) = os.path.split(filedir)
        # dir_name = 'datasets'
        # self.datasets_dir_path = os.path.join(projectdir, dir_name)
        self.datasets_dir_path = '/mnt/nariz/cazevedo/'

    def get_dataset(self, dataset, missing_mechanisms, missing_rates):
        print('Retrieving dataset...   ')
        custom_dataset = get_datasets.dataset_folder(dataset=dataset,
                                                     miss_strats=missing_mechanisms,
                                                     miss_rates=missing_rates,
                                                     n=len(missing_mechanisms))

        mnist_orig_ds = custom_dataset.orig_ds  # get the original dataset with its partitions
        mnist_miss_masks = custom_dataset.miss_masks  # get the miss masks for the n folds
        mnist_corr_ds = custom_dataset.ds_corruptor()  # get the corrupted datasets from the mask matrixes

        config_idx = 0
        incomplete_dataset = mnist_corr_ds['corr_X'][config_idx]['train_X']
        mask = mnist_miss_masks[config_idx]['train_X']
        original_dataset = mnist_orig_ds['train_X']

        return incomplete_dataset, mask, original_dataset

    def save_dataset(self, reconstructed_dataset, dataset_name, approach_name, missing_mechanism, missing_rate, run_index, alpha):
        # save the reconstructed dataset in datasets folder
        fn_fmt = "/{dataset}_{approach}_{missing_mechanism}_{missing_ratio}_{alpha}_{run_index}.pkl"
        file_name = fn_fmt.format(
            dataset=dataset_name,
            approach=approach_name,
            missing_mechanism=missing_mechanism,
            missing_ratio=missing_rate,
            alpha=alpha,
            run_index=run_index
        )
        path_file_name = self.datasets_dir_path + file_name

        print("Saving reconstructed dataset in path : ", path_file_name)
        reconstructed_dataset.to_pickle(path_file_name)

    def run(self, num_runs):
        datasets = self.config.get("Dataset")
        missing_mechanisms = self.config.get("MissingnessMechanism")
        missing_rates = self.config.get("MissingRate")
        dataset_name = datasets[0]
        approach_name = 'GAINCredit'
        missing_mechanism = 'MCAR'

        alphas = [0.1, 0.5, 1, 2, 10]

        for missing_rate in missing_rates:
            incomplete_dataset, mask, original_dataset = self.get_dataset(dataset_name, missing_mechanism,
                                                                          missing_rate)
            for alpha in alphas:
                for run_index in range(num_runs):
                    print("------------------------------------")
                    print("Run number : ", run_index)
                    print("Missing mechanism : ", missing_mechanism)
                    print("Missing rate : ", missing_rate)
                    print("Alpha : ", alpha)

                    gain.train(incomplete_dataset, mask, alpha)
                    reconstructed_dataset = gain.reconstruct(incomplete_dataset, mask, alpha)

                    self. save_dataset(reconstructed_dataset, dataset_name, approach_name, missing_mechanism, missing_rate, run_index, alpha)


if __name__ == "__main__":
    reconstruct_obj = ReconstructDataset()
    n_runs = 5
    reconstruct_obj.run(n_runs)