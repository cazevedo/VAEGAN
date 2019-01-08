import pickle
import numpy as np

def nrmse(original_dataset, reconstructed_dataset):
    """
    input:
        - original dataset: pandas dataframe
        - reconstructed dataset: pandas dataframe
    output:
        - NRMSE between the 2 datasets
    """
    x = original_dataset[0].values
    r = reconstructed_dataset[0].values
    
    x_max = x.max()
    x_min = x.min()
    
    missing_ratio = reconstructed_dataset[4]

    # TODO size of missing mask??  
    print(r.shape)
    n = r.shape[0]*r.shape[1]*missing_ratio # number of missing/imputed observations
   
    nrmse = np.sqrt(np.sum(np.square(x-r))/n)/(x_max-x_min)
    return nrmse

def evaluate_approach(original_dataset, reconstructed_datasets):
    """
    input:
        - original dataset: pandas dataframe
        - list of reconstructed dataset: list of pandas dataframes
    output:
        - mean of nrmse
        - variance of nrmse
    
    """
    
    nrmses = [nrmse(original_dataset, r) for r in reconstructed_datasets]
    
    return nrmses, np.mean(nrmses), np.std(nrmses)
    
    
orig = pickle.load(open("MNIST.pkl", "rb"))

# reconstructed = [pickle.load(open("MNIST_MeanImputation_MCAR_0.1_0.pkl", "rb"))]

#reconstructed.append(pickle.load(open("MNIST_MostFrequent_MCAR_0.1_0.pkl", "rb")))


#reconstructed = []
#for i in range(2):
#        reconstructed.append(pickle.load(open("MNIST_mean_MCAR_0.5_"+str(i)+".pkl", "rb")))
        
#evaluate_approach(orig, reconstructed)