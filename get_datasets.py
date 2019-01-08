import pandas as pd
import numpy as np
from torchvision import datasets as tds
import random
import copy
import time
import os

class dataset_folder():
    def __init__(self,dataset,miss_strats,miss_rates,n=1,target=None,train_ratio=None):
        self.config_file=self.config(miss_strats,miss_rates,n,train_ratio)
        self.orig_ds=self.ds_loader(dataset,target)
        self.miss_masks=self.make_miss_masks()
    
    #Creates config file with list of miss_strats and miss_rates of size n
    def config(self,miss_strats,miss_rates,n,train_ratio):
        miss_strats,miss_rates,train_ratio=self.verify_inputs(miss_strats,miss_rates,n,train_ratio)
        #miss_strats and miss_rates are now a lists of size n
        config_file={
                'n':n, #number of amputation rounds
                'miss_strats':miss_strats, #list of miss strats, 1 for each amputation round
                'miss_rates':miss_rates, #list of miss rates, 1 for each amputation round
                'train_ratio':train_ratio #train split ratio for the dataset
                }
        return config_file

    #Verifies all configuration inputs
    def verify_inputs(self,miss_strats,miss_rates,n,train_ratio):
        ### List of accepted strats
        accepted_strats=['MCAR','MAR']
        
        ## Verifying inputs
        ### Verify n
        if type(n)!=int:
            raise TypeError("The argument number of amputation rounds 'n' must be an integer")
        elif n<=0:
            raise ValueError("The argument number of amputation rounds 'n' must be bigger than 0")
            
            
        ### Verify train_ratio
        if train_ratio is None:
            train_ratio='default'
        elif type(train_ratio)!=float and type(train_ratio)!=int:
            raise TypeError("The argument number of imputation tests 'train_ratio' must be a float or an integer")
        elif train_ratio<=0 or train_ratio>1:
                raise ValueError("Valid 'train_ratio' splits are floats contained in ]0,1]")
        else:
            print('Using a {0:.01f}% train split'.format(round(train_ratio*100,2)))

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
                print('Length of given strategies does not match n. \nUsing last given strategy, {}, for remaining runs.'.format(miss_strats[-1]))
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
                print('Length of given missing rates does not match n. \nUsing last given rate, {}, for remaining runs.'.format(miss_rates[-1]))
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
    
        return miss_strats,miss_rates,train_ratio
    
    #Devolve orig_ds, um dicionário com todos os dataframes correspondentes
    def ds_loader(self,dataset,target):
        supported_datasets={'MNIST':self.load_MNIST,
                'credit':self.load_credit
                }

        if type(dataset)==pd.core.frame.DataFrame:
            if self.config_file['train_ratio']=='default':
                raise AttributeError("Please provide custom datasets with a train split ratio.")
            print("Sorry, personal dataframe processing is still under development!")
            return
        if type(dataset)==str:
            if dataset in supported_datasets:
                train_X,test_X,dtypes,train_target,test_target=supported_datasets[dataset]()
            else:
                raise NotImplementedError("{} not a supported dataset {}".format(dataset, list(supported_datasets.keys())))
                
        orig_ds={
                'train_X':train_X,
                'test_X':test_X,
                'dtypes':dtypes,
                'train_target':train_target,
                'test_target':test_target
                }
        return orig_ds
    
    ##
    def load_MNIST(self, normalized=True):
        script_dir = os.path.abspath(__file__)
        (filedir, tail) = os.path.split(script_dir)
        filename = 'datasets/MNIST'
        abs_file_path_to_mnist = os.path.join(filedir, filename)

        ##Create data partitions
        train_part=[True,False]#tds modes for calling MNIST default partitions
        train_ratio=self.config_file['train_ratio']
        
        #Creating  default partitions
        for mode in train_part:
            ds_mnist=tds.MNIST(root=abs_file_path_to_mnist,train=mode,download=True)
            if mode: #Train partition selected
                X,target=ds_mnist.train_data[:],ds_mnist.train_labels[:]
                train_X=pd.DataFrame(np.array([np.asarray(i).flatten() for i in X]),dtype=np.uint8)
                train_target=pd.DataFrame(np.array([np.asarray(i).flatten() for i in target]),columns=['target'],dtype=np.uint8).astype('category')
            else: #Test partition
                X,target=ds_mnist.test_data[:],ds_mnist.test_labels[:]
                test_X=pd.DataFrame(np.array([np.asarray(i).flatten() for i in X]),dtype=np.uint8)
                test_target=pd.DataFrame(np.array([np.asarray(i).flatten() for i in target]),columns=['target'],dtype=np.uint8).astype('category')         
        
        #If a different train_ratio is specified
        if train_ratio!='default':
            print('Leave train_ratio as blank for the common value used in benchmarks')
            #Fuse observations with new indexes before splitting them
            X=pd.concat([train_X,test_X],axis=0).reset_index(drop=True)
            target=pd.concat([train_target,test_target],axis=0).reset_index(drop=True)
            
            #Sample random test and train indexes, according to train_ratio
            test_index=random.sample(range(len(target)),int(len(target)*(1-train_ratio)))
            test_index.sort()
            train_index=list(set(range(len(target))).difference(test_index))
            
            #Split partitions
            train_X=X.loc[train_index]
            test_X=X.loc[test_index]
            train_target=target.loc[train_index]
            test_target=target.loc[test_index]
        
        #Save column datatypes
        dtypes=pd.concat([test_X,test_target],axis=1).dtypes
        for (index,value) in dtypes.items():
            if value=='uint8':
                dtypes.loc[index]='count'
            else:
                dtypes.loc[index]='cat'

        # Normalize
        train_X = train_X / 255.0
        test_X = test_X / 255.0

        return train_X,test_X,dtypes,train_target,test_target
    
    def load_credit(self):
        script_dir = os.path.abspath(__file__)
        (filedir, tail) = os.path.split(script_dir)
        filename = 'datasets/default of credit card clients.xls'
        abs_file_path = os.path.join(filedir, filename)

        #Import the dataset from excel
        raw=pd.read_excel(abs_file_path,
                          index_col='ID',
                          dtype='int64')
        
        train_ratio=self.config_file['train_ratio']
        
        ###Column type mapping
        #ordinal: belong to an ordered finite set -> int8
        #cat: belong to an unordered finite set -> category
        #real: take values in the real line R -> int64 or float64
        #target: predicted variable -> category
        #pos: real positive numbers -> uint64
        #count: positive ordered numbers -> uint16
        variables={'ordinal':['PAY_1','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6'],
                   'count':['AGE'],
                   'pos':['LIMIT_BAL','PAY_AMT1','PAY_AMT2','PAY_AMT3',
                          'PAY_AMT4','PAY_AMT5','PAY_AMT6'],
                    'cat':['default next month','SEX','EDUCATION','MARRIAGE'],
                    'target':'default next month',
                    'real':['BILL_AMT1','BILL_AMT2','BILL_AMT3',
                                 'BILL_AMT4','BILL_AMT5','BILL_AMT6',]
                    }
        #Format dtypes
        raw[variables['ordinal']]=raw[variables['ordinal']].astype(np.int8)
        raw[variables['count']]=raw[variables['count']].astype('uint8')
        raw[variables['pos']]=raw[variables['pos']].astype('uint64')
        raw[variables['cat']]=raw[variables['cat']].astype('category')
        raw[variables['real']]=raw[variables['real']].astype('int64')
        
        #Define train_ratio
        if train_ratio=='default':
            train_ratio=0.9 #Default ratio of 0.9 for this dataset
            
        #Split dataset - create indices
        # test_index=random.sample(range(len(raw)),int(len(raw)*(1-train_ratio)))
        # test_index.sort()
        # train_index=list(set(range(len(raw))).difference(test_index))

        # Force not random
        test_sep = int(len(raw)*train_ratio)
        train_index = list(range(0, test_sep))
        test_index = list(range(test_sep, len(raw)))

        #splits
        train_X=raw.loc[train_index, raw.columns != variables['target']]
        test_X=raw.loc[test_index, raw.columns != variables['target']]
        train_target=raw.loc[train_index, raw.columns==variables['target']]
        test_target=raw.loc[test_index, raw.columns==variables['target']]

        train_X = train_X[ ['LIMIT_BAL','PAY_AMT1','PAY_AMT2','PAY_AMT3', 'PAY_AMT4','PAY_AMT5','PAY_AMT6', 'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'] ]
        test_X = test_X[ ['LIMIT_BAL','PAY_AMT1','PAY_AMT2','PAY_AMT3', 'PAY_AMT4','PAY_AMT5','PAY_AMT6', 'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6'] ]

        def normalize(df):
            return (df-df.min())/(df.max()-df.min())

        train_X = normalize(train_X)
        test_X = normalize(test_X)

        #get dtypes series
        dtypes=raw.dtypes
        for (key, value) in variables.items():
            if key == 'target':
                continue
            dtypes[value]=key

        return train_X,test_X,dtypes,train_target,test_target
    
    #Receives orig_ds as an argument and returns encoded_ds
    ##same style as orig_ds but with encoded ordinal and categorical vars
    @staticmethod
    def encode_vars(orig_ds):
        encoded_ds={'dtypes':orig_ds['dtypes']}
        ds_dic={'target':pd.concat([orig_ds['test_target'],orig_ds['train_target']]),
                'X':pd.concat([orig_ds['test_X'],orig_ds['train_X']])}
        
        ## turns ordinal and count variables of the original dataframe to thermometer encodes
        def thermo_encode(col):
            placeholder=copy.deepcopy(col)
            
            #transforms values of the matrix to thermo_encoded vectors
            def encode_val(val):
                vector=np.zeros(max_v-min_v)
                if val!=min_v:
                    vector[:val-min_v]=1
                return vector
            
            #creates a dataframe made of thermo_encoded vectors for each column
            min_v=placeholder.min()
            max_v=placeholder.max()
            thermo_encode=pd.DataFrame([encode_val(val) for val in placeholder],
                                       columns=pd.MultiIndex.from_arrays([[placeholder.name]*(max_v-min_v),list(range((max_v-min_v)))]))
            print("{} column was thermometer encoded into {} columns.".format(col.name,thermo_encode.shape[1]))
            return thermo_encode
        
                ## turns ordinal and count variables of the original dataframe to thermometer encodes
        def onehot_encode(col):
            placeholder=copy.deepcopy(col)
            onehot_encode=pd.get_dummies(placeholder)
            onehot_encode.columns=pd.MultiIndex.from_arrays([[placeholder.name]*len(placeholder.cat.categories), list(placeholder.cat.categories)])
            print("{} column was one-hot encoded into {} columns.".format(placeholder.name,onehot_encode.shape[1]))
            return onehot_encode
        
        ## 1)Iterate X and target
        ## 2)Iterate columns and run encoders on ordinal and cat columns
        ## 3)Concat encoded columns as (var_name,col_nr)
        ## 3)Concat other columns and create multiindex of type (var_name,var_name)
        ## 4)Insert in encoded_ds
        for (name,df) in ds_dic.items():
            df_encoded=pd.DataFrame()
            for col in df:
                if encoded_ds['dtypes'][col]=='ordinal':
                    df_encoded=pd.concat([df_encoded,thermo_encode(df[col])],axis=1,levels=['variable','col_nr'])
                elif encoded_ds['dtypes'][col]=='cat':
                    df_encoded=pd.concat([df_encoded,onehot_encode(df[col])],axis=1)
            if name!='target':
                other_cols=df[encoded_ds['dtypes'].loc[(encoded_ds['dtypes'] != 'ordinal') & (encoded_ds['dtypes'] != 'cat')].index]
                other_cols.columns=pd.MultiIndex.from_arrays([other_cols.columns,other_cols.columns])
                df_encoded=pd.concat([df_encoded,other_cols],axis=1)
            if name=='X':
                encoded_ds['test_X']=df_encoded.loc[orig_ds['test_X'].index]
                encoded_ds['train_X']=df_encoded.loc[orig_ds['train_X'].index]
            else:
                encoded_ds['test_target']=df_encoded.loc[orig_ds['test_target'].index]
                encoded_ds['train_target']=df_encoded.loc[orig_ds['train_target'].index]
        return encoded_ds
    
    #Iterates over config_file and generates miss masks based on the miss_strats and miss_rates lists
    def make_miss_masks(self):
        # Map implemented miss strategy functions
        strategies={
                'MCAR':self.MCAR,
                'MAR':self.MAR}
        
        #Initialize miss masks dictionary
        miss_masks={}
        
        #Iterate over the n amputation rounds
        for i in range(self.config_file['n']):
            miss_masks[i]={
                    'train_X':None,
                     'test_X':None}
            for partition in miss_masks[i]:
                data=self.orig_ds[partition]
                length,width=data.shape
                miss_rate=self.config_file['miss_rates'][i]
                miss_strat=self.config_file["miss_strats"][i]
                #print(length,width,miss_rate)
                #print(strategies[miss_strat])
                mask=strategies[miss_strat](length,width,miss_rate)
                miss_masks[i][partition]=pd.DataFrame(mask,columns=data.columns,index=data.index) 
        return miss_masks
    
    #Generates a matrix of uniform distribution samples
    #truncates above miss_rate to 1 and below miss rate to 0
    def MCAR(self,length,width,miss_rate):
        #Create random matrix with given shape from uniform distribution in [0,1]
        mask_matrix=np.random.rand(length,width)
        #Random values below miss_rate get smoothed to 0
        mask_matrix[(mask_matrix<miss_rate)]=0
        #Random values above miss_rate get smoothed to 1
        mask_matrix[(mask_matrix>=miss_rate)]=1
        return mask_matrix
    
    # Applies Missing At Random to a ones_matrix with corresponding miss_rate
    def MAR(self,length,width,miss_rate):   
        ##Randomizing miss_spread
        #Percentage of 1's of the mask_matrix that are mixed with the 0's
        #a value of 1 would make the miss strategy MCAR, a value of 0 would yield
        #unfragmented blocks of 0's and 1's
        miss_spread=np.random.uniform(low=0.1, high=0.35)
        
        ##Initializing the mask matrix as a matrix of 1's
        mask_matrix=np.ones((length,width))
        #Number of missing values
        n_miss=int(mask_matrix.size*miss_rate)
        ##Create a mixture array
        #n_miss 0's and a certain number of 1's defined by miss_spread
        miss_array=np.array([0] * n_miss + [1] * int((mask_matrix.size-n_miss)*miss_spread))
        ##Shuffle the miss array
        np.random.shuffle(miss_array)
        
        #Iterate the mixture array and copy its values to the miss_matrix
        #Increment row wise, randomise new columnn at the lower edge of the matrix
        l=0
        cols=list(range(width))
        np.random.shuffle(cols)
        col=cols.pop(-1)
        for i in range(len(miss_array)):
            mask_matrix[l%length,col]=miss_array[i]
            l+=1
            if l==length:
                col=cols.pop(-1)
                l=0
        return mask_matrix
    
    #method that can be called to produce the corrupted datasets from the list of mask matrices
    def ds_corruptor(self):
        print('Generating the corrupted datasets from the mask matrixes.')
        #Initialize corrupted datasets dictionary
        corr_ds={
                'train_target':self.orig_ds['train_target'],
                 'test_target':self.orig_ds['test_target'],
                 'corr_X':{}}
        
        for i in self.miss_masks:
            corr_ds['corr_X'][i]={
                    'train_X':None,
                    'test_X':None}
            for partition in corr_ds['corr_X'][i]:
                mask_matrix=self.miss_masks[i][partition]
                dataframe=self.orig_ds[partition]
                # corr_ds['corr_X'][i][partition]=mask_matrix.where(mask_matrix==1,np.nan).mask(mask_matrix==1,dataframe)
                corr_ds['corr_X'][i][partition]=mask_matrix.where(mask_matrix==1,np.random.rand()).mask(mask_matrix==1,dataframe)
                # corr_ds['corr_X'][i][partition]=mask_matrix.where(mask_matrix==1, 0.5).mask(mask_matrix==1,dataframe)
        return corr_ds

####### Tests and example cals ######
        
def credit_example():
    credit=dataset_folder(dataset='credit',miss_strats=['MAR','MAR','MCAR'],miss_rates=0.5,n=2,train_ratio=0.9) 
    return credit

def mnist_example():
    mnist=dataset_folder(dataset='MNIST',miss_strats=['MAR','MAR','MCAR'],miss_rates=0.2,n=1)
    return mnist

#Class call example
def test():
    start_time = time.time()
    ##Credit dataset
    credit = credit_example()
    credit_orig_ds=credit.orig_ds #get the original dataset with its partitions
    credit_miss_masks=credit.miss_masks #get the miss masks for the n folds
    credit_corr_ds=credit.ds_corruptor() #get the corrupted datasets from the mask matrixes
    credit_encoded_ds=credit.encode_vars(credit_orig_ds)  
    print("Credit dataset example test completed in {:.2f} seconds.".format(time.time() - start_time))

    start_time = time.time()
    ##MNIST dataset
    mnist = mnist_example()
    mnist_orig_ds=mnist.orig_ds #get the original dataset with its partitions
    mnist_miss_masks=mnist.miss_masks #get the miss masks for the n folds
    mnist_corr_ds=mnist.ds_corruptor() #get the corrupted datasets from the mask matrixes
    mnist_encoded_ds=mnist.encode_vars(mnist_orig_ds)
    print("MNIST dataset example test completed in {:.2f} seconds.".format(time.time() - start_time))
    
if __name__ == "__main__":
     test()