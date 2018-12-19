import json
import os
import importlib.util
import numpy as np

### --------------------- DEBUG TOOLS -----------------------###
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm
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

    # Random sample generator for Z
    def sample_Z(m, n):
        return np.random.uniform(0., 1., size=[m, n])

    p_miss = 0.5
    dataset = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    if mode=='train':
        # ---- TRAIN ---- #
        Train_No = n_samples
        original_dataset, _ = dataset.train.next_batch(Train_No)
        mask = sample_M(Train_No, np.shape(original_dataset)[1], p_miss)

        (datasetLen, Dim) = np.shape(original_dataset)
        incomplete_dataset = np.zeros((datasetLen, Dim))
        # generate corrupted dataset
        for it in tqdm(range(datasetLen)):
            mb_idx = [it]
            X_mb = original_dataset[mb_idx, :]
            M_mb = mask[mb_idx, :]
            Z_mb = sample_Z(1, Dim)
            incomplete_dataset[it] = M_mb * X_mb + (1 - M_mb) * Z_mb

    elif mode == 'eval':
        # ---- TEST ---- #
        Test_No = n_samples
        original_dataset, _ = dataset.test.next_batch(Test_No)
        mask = sample_M(Test_No, np.shape(original_dataset)[1], p_miss)

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

def run():
    # Load json config file
    script_dir = os.path.abspath(__file__)
    (filedir, tail) = os.path.split(script_dir)
    (projectdir, tail) = os.path.split(filedir)
    filename = 'config.json'
    abs_file_path = os.path.join(projectdir, filename)

    with open(abs_file_path) as f:
        config = json.load(f)

    # get the path of the approaches to be compared
    approachApath = config.get("ApproachA")
    approachBpath = config.get("ApproachB")
    datasetPath = config.get("IncompleteDataset")
    mode = config.get("Mode")

    # execute the first approach
    spec = importlib.util.spec_from_file_location("appA", approachApath)
    appA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(appA)

    # execute the second approach
    # spec = importlib.util.spec_from_file_location("appB", approachBpath)
    # appB = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(appB)

    # TODO read the dataset from datasetPath

    if mode == 'train':
        original_dataset, incomplete_dataset, mask = get_dataset(mode, n_samples=20000)
        appA.train(incomplete_dataset, mask)
    elif mode == 'eval':
        original_dataset, incomplete_dataset, mask= get_dataset(mode, n_samples=100)
        reconstructed_dataset = appA.eval(incomplete_dataset, mask)

        for i in tqdm(range(len(incomplete_dataset))):
            inc = incomplete_dataset[i]
            rec = reconstructed_dataset[i]
            orig = original_dataset[i]

            samples = np.vstack([inc, rec, orig])
            fig = plot(samples)
            plt.savefig('Multiple_Impute_out1/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
            plt.close(fig)

if __name__ == "__main__":
    run()