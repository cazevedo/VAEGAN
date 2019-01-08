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

    def run(self, run_index=0):
        datasets = self.config.get("Dataset")
        missing_mechanisms = self.config.get("MissingnessMechanism")
        missing_rates = self.config.get("MissingRate")

        r_datasets_paths = [] # list where all the paths of the reconstructed datasets will be stored

        for dataset in datasets:
            print('Retrieving dataset...   ')
            custom_dataset = get_datasets.dataset_folder(dataset=dataset,
                                                         miss_strats=missing_mechanisms,
                                                         miss_rates=missing_rates,
                                                         n=len(missing_mechanisms))

            for config_idx in range(len(missing_mechanisms)):
                mnist_orig_ds = custom_dataset.orig_ds  # get the original dataset with its partitions
                mnist_miss_masks = custom_dataset.miss_masks  # get the miss masks for the n folds
                mnist_corr_ds = custom_dataset.ds_corruptor()  # get the corrupted datasets from the mask matrixes

                incomplete_dataset = mnist_corr_ds['corr_X'][config_idx]['train_X']
                mask = mnist_miss_masks[config_idx]['train_X']
                original_dataset = mnist_orig_ds['train_X']

                # Reconstruct the dataset for each approach
                approaches = self.config.get("Approaches")
                approaches_mode = self.config.get("ApproachesMode")
                for app_index, app_path in enumerate(approaches):
                    # get the path of the approach to be compared
                    spec = importlib.util.spec_from_file_location("approach", app_path)
                    appch = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(appch)

                    if approaches_mode[app_index] == 'train':
                        appch.train(incomplete_dataset, mask)
                        reconstructed_dataset = appch.reconstruct(incomplete_dataset, mask)
                    elif approaches_mode[app_index] == 'reconstruct':
                        reconstructed_dataset = appch.reconstruct(incomplete_dataset, mask)

                    # save the reconstructed dataset in datasets folder
                    fn_fmt = "/{dataset}_{approach}_{missing_mechanism}_{missing_ratio}_{run_index}.pkl"
                    # get the approach name
                    (filedir, app_name) = os.path.split(app_path)
                    app_name = app_name.split('.')[0]
                    file_name = fn_fmt.format(
                        dataset=dataset,
                        approach=app_name,
                        missing_mechanism=missing_mechanisms[config_idx],
                        missing_ratio=missing_rates[config_idx],
                        run_index=run_index
                    )
                    path_file_name = self.datasets_dir_path+file_name

                    reconstructed_dataset.to_pickle(path_file_name)
                    r_datasets_paths.append(path_file_name)

                    # # Print for DEBUG ##
                    # for i in tqdm(range(len(reconstructed_dataset))):
                    #     if i % 40000 == 0:
                    #         inc = incomplete_dataset.loc[i].values
                    #         rec = reconstructed_dataset.loc[i].values
                    #         orig = original_dataset.loc[i].values
                    #
                    #         samples = np.vstack([inc, rec, orig])
                    #         fig = plot(samples)
                    #         plt.savefig('Multiple_Impute_out1/{}_sample_{}.png'.format(file_name.split('.pkl')[0],str(i)), bbox_inches='tight')
                    #         plt.close(fig)
                    # ## ----- ##

                #Reconstruct the dataset for each baseline method
                baseline = self.config.get("BaselineMethods")
                for method in baseline:
                    method_path = self.filedir+'/'+method+'.py'
                    spec = importlib.util.spec_from_file_location("method", method_path)
                    mthd = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mthd)

                    reconstructed_dataset = mthd.reconstruct(custom_dataset, config_idx)

                    # save the reconstructed dataset in datasets folder
                    fn_fmt = "/{dataset}_{approach}_{missing_mechanism}_{missing_ratio}_{run_index}.pkl"
                    file_name = fn_fmt.format(
                        dataset=dataset,
                        approach=method,
                        missing_mechanism=missing_mechanisms[config_idx],
                        missing_ratio=missing_rates[config_idx],
                        run_index=run_index
                    )
                    path_file_name = self.datasets_dir_path + file_name

                    reconstructed_dataset.to_pickle(path_file_name)
                    r_datasets_paths.append(path_file_name)

                    # ## Print for DEBUG ##
                    # for i in tqdm(range(len(reconstructed_dataset))):
                    #     if i % 40000 == 0:
                    #         inc = incomplete_dataset.loc[i].values
                    #         rec = reconstructed_dataset.loc[i].values
                    #         orig = original_dataset.loc[i].values
                    #
                    #         samples = np.vstack([inc, rec, orig])
                    #         fig = plot(samples)
                    #         plt.savefig('Multiple_Impute_out1/{}_sample_{}.png'.format(file_name.split('.pkl')[0],str(i)), bbox_inches='tight')
                    #         plt.close(fig)
                    # ## ----- ##

        return r_datasets_paths

if __name__ == "__main__":
    reconstruct_obj = ReconstructDataset()
    n_runs = 20

    full_dataset_paths = []
    for run_number in range(n_runs):
        r_datasets_paths = reconstruct_obj.run(run_number)
        full_dataset_paths.append(r_datasets_paths)
        print(r_datasets_paths)

    print(full_dataset_paths)