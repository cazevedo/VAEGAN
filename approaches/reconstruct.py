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

### --------------------- DEBUG TOOLS -----------------------###
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import pandas as pd

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

def get_dataset(mode, n_samples):
    def sample_M(m, n, p):
        A = np.random.uniform(0., 1., size=[m, n])
        B = A > p
        C = 1. * B
        return C

    def sample_MNAR(original_dataset):
        mask = np.ones(np.shape(original_dataset))
        mask = mask.reshape((np.shape(original_dataset)[0], 28, 28))

        # missing_rows = [3, 22]
        missing_rows = [3, 12, 18, 22, 23]
        for i in range(np.shape(original_dataset)[0]):
            mask[i][missing_rows, :] = mask[i][missing_rows, :] * 0
        return mask.reshape((np.shape(original_dataset)[0], 28*28))

    # Random sample generator for Z
    def sample_Z(m, n):
        return np.random.uniform(0., 1., size=[m, n])

    p_miss = 0.2
    dataset = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    original_dataset, _ = dataset.train.next_batch(n_samples)

    if mode == 'MCAR':
        mask = sample_M(n_samples, np.shape(original_dataset)[1], p_miss)
    elif mode == 'MNAR':
        mask = sample_MNAR(original_dataset)

    (datasetLen, Dim) = np.shape(original_dataset)
    incomplete_dataset = np.zeros((datasetLen, Dim))
    # generate corrupted dataset
    for it in tqdm(range(datasetLen)):
        mb_idx = [it]
        X_mb = original_dataset[mb_idx, :]
        M_mb = mask[mb_idx, :]
        Z_mb = sample_Z(1, Dim)
        incomplete_dataset[it] = M_mb * X_mb + (1 - M_mb) * Z_mb

    original_dataset = pd.DataFrame(original_dataset)
    incomplete_dataset = pd.DataFrame(incomplete_dataset)
    mask = pd.DataFrame(mask)

    return  original_dataset, incomplete_dataset, mask
### ---------------------  -----------------------###
### ---------------------  -----------------------###

class ReconstructDataset(object):
    def __init__(self):
        # Load json config file (should be in the root dir of the project)
        script_dir = os.path.abspath(__file__)
        (self.filedir, tail) = os.path.split(script_dir)
        (projectdir, tail) = os.path.split(self.filedir)
        filename = 'config.json'
        abs_file_path = os.path.join(projectdir, filename)

        with open(abs_file_path) as f:
            self.config = json.load(f)

        script_dir = os.path.abspath(__file__)
        (filedir, tail) = os.path.split(script_dir)
        (projectdir, tail) = os.path.split(filedir)
        dir_name = 'datasets'
        self.datasets_dir_path = os.path.join(projectdir, dir_name)

    def run(self):
        # TODO change this for Francisco's implementation
        # original_dataset, incomplete_dataset, mask = get_dataset(mode='MCAR', n_samples=500)
    
        mnist = get_datasets.mnist_example()
        mnist_miss_masks = mnist.miss_masks  # get the miss masks for the n folds
        mnist_corr_ds = mnist.ds_corruptor()  # get the corrupted datasets from the mask matrixes

        config_idx = 0
        incomplete_dataset = mnist_corr_ds['corr_X'][config_idx]['test_X']
        mask = mnist_miss_masks[config_idx]['test_X']

        # print(np.shape(incomplete_dataset_2))
        # print(incomplete_dataset_2)
        #
        # print(np.shape(incomplete_dataset))
        # print(incomplete_dataset)
        #
        # print(np.shape(mask))
        # print(mask)
        #
        # sys.exit(0)

        r_datasets_paths = []
        # Reconstruct the dataset for each approach
        n_approaches = self.config.get("NumberOfApproaches")
        for i in range(n_approaches):
            # get the path of the approach to be compared
            approach_path = self.config.get("Approach"+str(i+1))
            spec = importlib.util.spec_from_file_location("approach", approach_path)
            appch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(appch)

            # execution mode
            mode = self.config.get("Approach"+str(i+1)+"Mode")

            if mode == 'train':
                appch.train(incomplete_dataset, mask)
                reconstructed_dataset = appch.reconstruct(incomplete_dataset, mask)
            elif mode == 'reconstruct':
                reconstructed_dataset = appch.reconstruct(incomplete_dataset, mask)

            # save the reconstructed dataset in datasets folder
            file_name = self.datasets_dir_path+"/Approach"+str(i+1)+"_rd"
            reconstructed_dataset.to_pickle(file_name)
            r_datasets_paths.append(file_name)

            ## Print for DEBUG ##
            for i in tqdm(range(len(reconstructed_dataset))):
                if i % 1000 == 0:
                    inc = incomplete_dataset.loc[i].values
                    rec = reconstructed_dataset.loc[i].values
                    orig = original_dataset.loc[i].values

                    samples = np.vstack([inc, rec, orig])
                    fig = plot(samples)
                    plt.savefig('Multiple_Impute_out1/app{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)
            ## ----- ##

        # Reconstruct the dataset for each baseline method
        baseline = self.config.get("BaselineMethods")
        for method in baseline:
            method_path = self.filedir+'/'+method+'.py'
            spec = importlib.util.spec_from_file_location("method", method_path)
            mthd = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mthd)

            reconstructed_dataset = mthd.reconstruct(incomplete_dataset, mask)

            # save the reconstructed dataset in datasets folder
            file_name = self.datasets_dir_path+'/'+method+'_rd'
            reconstructed_dataset.to_pickle(file_name)
            r_datasets_paths.append(file_name)

            ## Print for DEBUG ##
            for i in tqdm(range(len(reconstructed_dataset))):
                if i % 1000 == 0:
                    inc = incomplete_dataset.loc[i].values
                    rec = reconstructed_dataset.loc[i].values
                    orig = original_dataset.loc[i].values

                    samples = np.vstack([inc, rec, orig])
                    fig = plot(samples)
                    plt.savefig('Multiple_Impute_out1/{}{}.png'.format(str(i).zfill(3), method), bbox_inches='tight')
                    plt.close(fig)
            ## ----- ##

        return r_datasets_paths

if __name__ == "__main__":
    reconstruct_obj = ReconstructDataset()
    r_datasets_paths = reconstruct_obj.run()
    print(r_datasets_paths)