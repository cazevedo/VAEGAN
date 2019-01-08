"""
Comparing runtimes for generating the MNIST Pandas dataframe or NumPy array

"""

import pandas as pd
import numpy as np
import torchvision.datasets as ds #using torchvision's MNIST loader
import time

ds_mnist=ds.MNIST(root='VAEGAN/supported_datasets/MNIST',download=True)

##
#NumPy array to mnist

start_time = time.time()
np_ds=np.array([np.array(i).flatten() for i in ds_mnist.train_data[:]])
print(time.time() - start_time)

print(np.shape(np_ds))

## ~ 0,7 seconds in my machine

##
#Pandas dataframe, the wrong way

# start_time = time.time()
# pd_ds=pd.DataFrame((np.asarray(i).flatten() for i in ds_mnist.train_data[:]),dtype=np.uint8)
# print(time.time() - start_time) ## ~ 250 seconds in my machine!

# print(np.shape(pd_ds))

##
#Pandas dataframe, the right way

start_time = time.time()
pd_ds=pd.DataFrame(np.array([np.asarray(i).flatten() for i in ds_mnist.train_data[:]]),dtype=np.uint8)
print(time.time() - start_time) ## ~ 0,7 seconds in my machine

print(np.shape(pd_ds))


print(np.shape(ds_mnist.train_data))

start_time = time.time()
pd_ds=pd.DataFrame(ds_mnist.train_data)
print(time.time() - start_time) ## ~ 0,7 seconds in my machine

print(np.shape(pd_ds))


### CONCLUSIONS
## Pandas didn't do a good job getting data from a NumPy array generator.
## Does an equivalent job when it gets data from a unique np.array built from the same generator.
