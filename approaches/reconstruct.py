import json
import os
import importlib.util

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

def sample_M(m, n, p):
    A = np.random.uniform(0., 1., size=[m, n])
    B = A > p
    C = 1. * B
    return C

def run():
    script_dir = os.path.abspath(__file__)
    (filedir, tail) = os.path.split(script_dir)
    (projectdir, tail) = os.path.split(filedir)
    filename = 'config.json'
    abs_file_path = os.path.join(projectdir, filename)

    with open(abs_file_path) as f:
        data = json.load(f)

    # get the path of the approaches to be compared
    approachApath = data.get("ApproachA")
    approachBpath = data.get("ApproachB")
    datasetPath = data.get("OriginalDataset")
    mode = data.get("Mode")

    # execute the first approach
    spec = importlib.util.spec_from_file_location("appA", approachApath)
    appA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(appA)

    # execute the second approach
    # spec = importlib.util.spec_from_file_location("appB", approachBpath)
    # appB = importlib.util.module_from_spec(spec)
    # spec.loader.exec_module(appB)

    # TODO read the dataset from datasetPath
    #### ---------------------------------------------------------------------- ###
    p_miss = 0.5
    Train_No = 10
    Test_No = 10
    dataset = input_data.read_data_sets('../../MNIST_data', one_hot=True)

    trainX, _ = dataset.train.next_batch(Train_No)
    trainMask = sample_M(Train_No, np.shape(trainX)[1], p_miss)

    testX, _ = dataset.test.next_batch(Test_No)
    testMask = sample_M(Test_No, np.shape(testX)[1], p_miss)
    #### ---------------------------------------------------------------------- ###

    if mode == 'train':
        appA.train(trainX, trainMask)
    elif mode == 'eval':
        reconstructed_dataset = appA.eval(testX, testMask)

if __name__ == "__main__":
    run()