import json
import os
import importlib.util
import numpy as np
from tqdm import tqdm

### --------------------- DEBUG TOOLS -----------------------###
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

        missing_rows = [3, 12, 18, 22, 23]
        for i in range(np.shape(original_dataset)[0]):
            mask[i][missing_rows, :] = mask[i][missing_rows, :] * 0
        return mask.reshape((np.shape(original_dataset)[0], 28*28))

    # Random sample generator for Z
    def sample_Z(m, n):
        return np.random.uniform(0., 1., size=[m, n])

    p_miss = 0.5
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

    return  original_dataset, incomplete_dataset, mask
### ---------------------  -----------------------###
### ---------------------  -----------------------###

class ReconstructDataset(object):
    def __init__(self):
        # Load json config file (should be in the root dir of the project)
        script_dir = os.path.abspath(__file__)
        (filedir, tail) = os.path.split(script_dir)
        (projectdir, tail) = os.path.split(filedir)
        filename = 'config.json'
        abs_file_path = os.path.join(projectdir, filename)

        with open(abs_file_path) as f:
            self.config = json.load(f)

    def run(self):
        n_approaches = self.config.get("NumberOfApproaches")
        for i in range(n_approaches):
            # get the path of the approach to be compared
            approach_path = self.config.get("Approach"+str(i+1))
            spec = importlib.util.spec_from_file_location("approach", approach_path)
            appch = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(appch)

            # execution mode
            mode = self.config.get("Approach"+str(i+1)+"Mode")

            # TODO get the dataset from pandas dataframe and make each approach compliant
            original_dataset, incomplete_dataset, mask = get_dataset(mode='MNAR', n_samples=1000)

            if mode == 'train':
                appch.train(incomplete_dataset, mask)
                reconstructed_dataset = appch.eval(incomplete_dataset, mask)
            elif mode == 'eval':
                reconstructed_dataset = appch.eval(incomplete_dataset, mask)

            for i in tqdm(range(len(reconstructed_dataset))):
                if i % 100 == 0:
                    inc = incomplete_dataset[i]
                    rec = reconstructed_dataset[i]
                    orig = original_dataset[i]

                    samples = np.vstack([inc, rec, orig])
                    fig = plot(samples)
                    plt.savefig('Multiple_Impute_out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
                    plt.close(fig)

            return reconstructed_dataset

if __name__ == "__main__":
    reconstruct_obj = ReconstructDataset()
    reconstruct_obj.run()