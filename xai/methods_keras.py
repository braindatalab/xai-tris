import keras

from typing import List, Dict
from common import DataRecord, SEED

from innvestigate import create_analyzer
from innvestigate.utils import model_wo_softmax

# from scipy.ndimage import sobel, laplace
import numpy as np

from keras.layers import Conv2D

np.random.seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

from tensorflow import set_random_seed
set_random_seed(SEED)


def get_keras_attributions(model: keras.Sequential, dataset: DataRecord, methods: List, test_size: int, mini_batch: int) -> Dict:
    attributions = dict()
    n_dim = dataset.x_test.shape[1]
    edge_length = int(np.sqrt(n_dim))

    is_conv = False
    for layer in model.layers:
        if type(layer) == Conv2D:
            is_conv = True
    
    if is_conv:
        x_train = np.reshape(dataset.x_train.detach().numpy(), (dataset.x_train.shape[0],edge_length,edge_length,1))
        x_test = np.reshape(dataset.x_test[:test_size].detach().numpy(), (test_size,edge_length,edge_length,1))
    else:
        x_train = np.reshape(dataset.x_train.detach().numpy(), (dataset.x_train.shape[0],n_dim))
        x_test = np.reshape(dataset.x_test[:test_size].detach().numpy(), (test_size,n_dim))
    model_wo_soft = model_wo_softmax(model)
    
    for method_name in methods:
        print(f'Calculating attributions for {method_name}')
        analyzer = create_analyzer(method_name,        # analysis method identifier
                                    model_wo_soft, # model without softmax output
                                    **{})      # optional analysis parameters, defaulting to {}
        
        if method_name == 'pattern.net' or method_name == 'pattern.attribution':
            analyzer.fit(x_train)

        for layer in model.layers:
            if is_conv:
                method_attributions = np.zeros((dataset.x_test[:test_size].shape[0], edge_length, edge_length, 1))
            else:
                method_attributions = np.zeros((dataset.x_test[:test_size].shape[0], n_dim))
        for ind in range(0, test_size, mini_batch):
            method_attributions[ind:ind+mini_batch] = analyzer.analyze(x_test[ind:ind+mini_batch])
        attributions[method_name] = method_attributions        
    return attributions

# def get_filter_attributions(dataset: DataRecord, methods: List, test_size: int) -> Dict:
#     attributions = dict()
#     n_dim = dataset.x_test.shape[1]
#     edge_length = int(np.sqrt(n_dim))

#     filter_dict = {
#         'sobel': sobel,
#         'laplace': laplace,
#         'rand': rand_baseline,
#         'x': x_baseline
#     }
#     for method_name in methods:
#         print(f'Calculating attributions for {method_name}')
#         method_attributions = np.zeros((dataset.x_test[:test_size].shape[0], edge_length, edge_length))
#         for ind in range(test_size):
#             method_attributions[ind:ind+1] = filter_dict.get(method_name)(dataset.x_test[ind].float().detach().numpy().reshape(edge_length, edge_length))
#         attributions[method_name] = method_attributions


#     attributions['rand'] = np.random.uniform(low=-1.0, high=1.0, size=(dataset.x_test.shape[0], edge_length, edge_length))
#     attributions['x'] = np.apply_along_axis(sum_to_1, 1, dataset.x_test.float().detach().numpy())
#     return attributions

# Scale matrix to sum to 1
def sum_to_1(mat):
    return mat / np.sum(mat)

def rand_baseline(data: np.array):
    rands = np.random.uniform(low=-1.0, high=1.0, size=(data.shape))
    return sum_to_1(rands)

def x_baseline(data: np.array):
    return sum_to_1(data)
