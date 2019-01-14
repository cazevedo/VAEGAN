import pickle
import numpy as np
from sys import argv, exit

def get_original(filename):
    orig, _, _ = pickle.load(open(filename, "rb"))
    return orig

def get_reconstructed(filename):
    dataset_name, approach, mechanism, missing_ratio, n = filename.split('.pkl')[0].split('_')
    missing_ratio = float(missing_ratio)

    rec = pickle.load(open(filename, "rb"))

    return rec, missing_ratio    

def test():
    print('HEY')

def nrmse(original_dataset, reconstructed_dataset, missing_ratio):
    """
    input:
        - original dataset: pandas dataframe
        - reconstructed dataset: pandas dataframe
    output:
        - NRMSE between the 2 datasets
    """
    x = original_dataset.values
    r = reconstructed_dataset.values
    
    x_max = x.max()
    x_min = x.min()

    # TODO size of missing mask??  
    # print(r.shape)
    n = r.shape[0]*r.shape[1]*missing_ratio # number of missing/imputed observations
   
    nrmse = np.sqrt(np.sum(np.square(x-r))/n)/(x_max-x_min)
    return nrmse

def evaluate_approach(original_dataset, reconstructed_datasets):
    """
    input:
        - original dataset: string filename
        - list of reconstructed dataset: list of filenames
    output:
        - mean of nrmse
        - variance of nrmse

    """
    
    orig = get_original(original_dataset)

    nrmses = [nrmse(orig, *get_reconstructed(r)) for r in reconstructed_datasets]
    
    return nrmses, np.mean(nrmses), np.std(nrmses)


if __name__ == "__main__":

    if len(argv) < 3:
        print("Usage: {} ORIGINAL.pkl RECONSTRUCTED.pkl".format(argv[0]))
        exit()

    orig = get_original(argv[1])

    rec, missing_ratio = get_reconstructed(argv[2])

    stat = nrmse(orig, rec, missing_ratio)

    print("{} - NRMSE - {}".format(argv[2], stat))
