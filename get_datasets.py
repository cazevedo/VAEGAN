import pandas as pd
import numpy as np
from torchvision import datasets as tds
import random


class dataset_folder():
    def __init__(self,dataset,miss_strats,miss_rates,n,target=None,train_ratio=0.9):
        self.config=self.config(miss_strats,miss_rates,n)
        self.orig_ds=self.ds_loader(dataset,target,train_ratio)
        self.miss_masks=self.make_miss_masks()

    #Creates config file with list of miss_strats and miss_rates of size n
    def config(self,miss_strats,miss_rates,n):
        miss_strats,miss_rates=self.verify_inputs(miss_strats,miss_rates,n)
        #miss_strats and miss_rates are now a lists of size n
        config={
                'miss_strats':miss_strats,
                'miss_rates':miss_rates
                }
        return config

    #Verifies all configuration inputs
    def verify_inputs(self,miss_strats,miss_rates,n):
        ### List of accepted strats
        accepted_strats=['MCAR','MNAR']

        ## Verifying inputs
        ### Verify n
        if type(n)!=int:
            raise TypeError("The argument number of imputation tests 'n' must be an integer")

        ### Verify miss_strats
        if type(miss_strats)==list:
            if len(miss_strats)==n:
                for strat in miss_strats:
                    strat=str(strat)
                    if strat not in accepted_strats:
                        raise NotImplementedError("The strategy {} is not implemented. Accepted strategies are: {}".format(strat,accepted_strats))
            ### Else introduces a lazy feature. When the list is short, the last strategy is used
            ### for all remaining runs. Ex: n=3 and miss_strats[MCAR,MNAR]  -> [MCAR,MNAR,MNAR]
            else:
                for strat in miss_strats:
                    strat=str(strat)
                    if strat not in accepted_strats:
                        raise NotImplementedError("The strategy {} is not implemented. Accepted strategies are: {}".format(strat,accepted_strats))
                print('Length of given strategies does not match n. \nUsing last given strategy, {}, for remaining runs'.format(miss_strats[-1]))
                miss_strats.extend([strat]*(n-len(miss_strats)))
                miss_strats=miss_strats[:n] ##Cover case where miss_strat list size is bigger than n
        ### If only one valid strategy is used as a string then it is applied to all runs
        elif type(miss_strats)==str:
            if miss_strats not in accepted_strats:
                raise NotImplementedError("The strategy {} is not implemented. Accepted strategies are: {}".format(miss_strats,accepted_strats))
            miss_strats=[miss_strats]*n
        else:
            raise TypeError("The [miss_strats] argument must be a string or a list of elements contained in {}".format(accepted_strats))

        ### Verify miss_rates
        if type(miss_rates)==list:
            if len(miss_rates)==n:
                for rate in miss_rates:
                    if type(rate)!=float:
                        raise TypeError("Valid missing rates are floats contained in ]0,1[")
                    if (rate<=0) or (rate>=1):
                        raise ValueError("Valid missing rates are floats contained in ]0,1[")
            ### When the list is short, the last rate is used for all remaining runs.
            ### Ex: n=3 and miss_rates[0.1,0.2]  -> [0.1,0.2,0.2]
            else:
                for rate in miss_rates:
                    if type(rate)!=float:
                        raise TypeError("Valid missing rates are floats contained in ]0,1[")
                    if (rate<=0) or (rate>=1):
                        raise ValueError("Valid missing rates are floats contained in ]0,1[")
                print('Length of given missing rates does not match n. \nUsing last given rate, {}, for remaining runs'.format(miss_rates[-1]))
                miss_rates.extend([rate]*(n-len(miss_rates)))
                miss_rates=miss_rates[:n] ##Cover case where miss_rates list size is bigger than n
        ### If only one valid rate is given then it is applied to all runs
        elif type(miss_rates)==float:
            if type(miss_rates)!=float:
                raise TypeError("Valid missing rates are floats contained in ]0,1[")
            if (miss_rates<=0) or (miss_rates>=1):
                raise ValueError("Valid missing rates are floats contained in ]0,1[")
            miss_rates=[miss_rates]*n
        else:
            raise TypeError("The [miss_rates] argument must be a float or a list of elements contained in ]0,1[")

        return miss_strats,miss_rates

    #Devolve orig_ds, um dicionário com todos os dataframes correspondentes
    def ds_loader(self,dataset,target,train_ratio):
        supported_datasets={
                #'MNIST':load_MNIST(),
                'credit':self.load_credit(train_ratio)
                }

        if type(dataset)==pd.core.frame.DataFrame:
            print("Sorry, still under development!")
            return
        if type(dataset)==str:
            if dataset in supported_datasets:
                train_X,test_X,dtypes,train_target,test_target=supported_datasets[dataset]

        orig_ds={
                'train_X':train_X,
                'test_X':test_X,
                'dtypes':dtypes,
                'train_target':train_target,
                'test_target':test_target
                }
        return orig_ds

    ##
    def load_MNIST():
        return

    def load_credit(self, train_ratio):
        #Import the dataset from excel
        raw=pd.read_excel('datasets/default of credit card clients.xls',
                          index_col='ID',
                          dtype='int64')

        ###Column type mapping
        #ordinal: belong to an ordered finite set -> uint8
        #categorical: belong to an unordered finite set -> category
        #real: take values in the real line R -> int64 or float64
        #target: predicted variable -> category
        variables={'ordinal_vars':['AGE','PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'],
                    'categorical_vars':['default next month','SEX','EDUCATION','MARRIAGE'],
                    'target_var':'default next month',
                    'real_vars':['LIMIT_BAL','BILL_AMT1','BILL_AMT2','BILL_AMT3',
                                 'BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1',
                                 'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
                    }
        #Format dtypes
        raw[variables['ordinal_vars']]=raw[variables['ordinal_vars']].astype(np.uint8)
        raw[variables['categorical_vars']]=raw[variables['categorical_vars']].astype('category')
        raw[variables['real_vars']]=raw[variables['real_vars']].astype('int64')

        #Split dataset - create indices
        test_index=random.sample(range(len(raw)),int(len(raw)*(1-train_ratio)))
        test_index.sort()
        train_index=list(set(range(len(raw))).difference(test_index))
        #splits
        train_X=raw.loc[train_index, variables['target_var']]
        test_X=raw.loc[test_index, raw.columns != variables['target_var']]
        train_target=raw.loc[train_index, raw.columns==variables['target_var']]
        test_target=raw.loc[test_index, raw.columns==variables['target_var']]

        #get dtypes series
        dtypes=raw.dtypes

        return train_X,test_X,dtypes,train_target,test_target

    #Iterates over config_file and generates miss masks based on the miss_strats and miss_rates lists
    def make_miss_masks(self):
        print("Sorry, still under development!")
        return

    # Applies missing completely at random to a ones_matrix with corresponding miss_rate
    def MCAR(ones_matrix, miss_rate):
        return mask_matrix

    # Applies MNAR to a ones_matrix with corresponding miss_rate
    def MNAR(ones_matrix, miss_rate):
        return mask_matrix

    #method that can be called to produce the corrupted datasets from the list of mask matrices
    def ds_corruptor(self):
        return corrupted_datasets

def test():
    #Class call example
    credit=dataset_folder(dataset='credit',miss_strats='MCAR',miss_rates=0.5,n=3)
    return credit

if __name__ == "__main__":
    credit = test()

