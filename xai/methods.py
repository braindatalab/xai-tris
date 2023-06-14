import torch

# https://captum.ai/api/
from captum.attr import IntegratedGradients, Saliency, DeepLift, DeepLiftShap, GradientShap, GuidedBackprop,  GuidedGradCam, Deconvolution, ShapleyValueSampling, Lime, KernelShap, LRP, FeaturePermutation

from typing import List, Dict
from common import DataRecord, SEED

from scipy.ndimage import sobel, laplace
import numpy as np

from training.models import LLR, MLP, CNN, CNN8by8

np.random.seed(SEED)

from torch import manual_seed
manual_seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)


# mini_batch = 5

baselines_methods = ["Integrated Gradients", "DeepLift", "DeepSHAP", "Gradient SHAP", "Shapley Value Sampling", "Kernel SHAP", "LIME"]

def get_attributions(model: torch.nn.Module, dataset: DataRecord, methods: List, test_size: int, mini_batch: int) -> Dict:
    attributions = dict()
    n_dim = dataset.x_test.shape[1]
    edge_length = int(np.sqrt(n_dim)) # pixel width/height, 8x8 normally and 24x24 for the scaled up experiment

    for method_name in methods:
        if method_name not in methods_dict:
            raise Exception(f'Method "{method_name}" is either not (yet) implemented or misspelt!')
    
    if 'DeepSHAP' in methods or 'Gradient SHAP' in methods:
        # baselines = torch.mean(dataset.x_test, dim=0).repeat(mini_batch, 1)
        baselines = torch.zeros((mini_batch, n_dim))

    for method_name in methods:
        if method_name == 'Guided GradCAM' and type(model) != CNN and type(model) != CNN8by8:
            continue
        # print(f'Calculating attributions for {method_name}')
        if type(model) == CNN or type(model) == CNN8by8:
            method_attributions = np.zeros((test_size, 1, edge_length, edge_length))
        else:
            method_attributions = np.zeros((test_size, n_dim))
        print('Calculating attributions for method', method_name)
        for ind in range(0, test_size, mini_batch):
            if type(model) == CNN or type(model) == CNN8by8:
                x = dataset.x_test[ind:ind+mini_batch].reshape(dataset.x_test[ind:ind+mini_batch].shape[0], 1, edge_length, edge_length)
            else:
                x = dataset.x_test[ind:ind+mini_batch].reshape(dataset.x_test[ind:ind+mini_batch].shape[0], n_dim)
            x.requires_grad = True
            if method_name in baselines_methods:
                if type(model) == CNN or type(model) == CNN8by8:
                    baselines = baselines.reshape(mini_batch, 1, edge_length, edge_length)
                else:
                    baselines = baselines.reshape(mini_batch, n_dim)
                method_attributions[ind:ind+mini_batch] = methods_dict.get(method_name)(data=x.float(), target=dataset.y_test[ind:ind+mini_batch], model=model, baselines=baselines.float()).detach().numpy()
            else:
                method_attributions[ind:ind+mini_batch] = methods_dict.get(method_name)(data=x.float(), target=dataset.y_test[ind:ind+mini_batch], model=model).detach().numpy()
        attributions[method_name] = method_attributions
    return attributions

def get_baselines(dataset: DataRecord, methods: List, test_size: int) -> Dict:
    attributions = dict()
    n_dim = dataset.x_test.shape[1]
    edge_length = int(np.sqrt(n_dim))

    filter_dict = {
        'sobel': sobel,
        'laplace': laplace,
        'rand': rand_baseline,
        'x': x_baseline
    }
    for method_name in methods:
        print(f'Calculating attributions for {method_name}')
        method_attributions = np.zeros((dataset.x_test[:test_size].shape[0], edge_length, edge_length))
        for ind in range(test_size):
            method_attributions[ind:ind+1] = filter_dict.get(method_name)(dataset.x_test[ind].float().detach().numpy().reshape(edge_length, edge_length))
        attributions[method_name] = method_attributions


    attributions['rand'] = np.random.uniform(low=-1.0, high=1.0, size=(dataset.x_test.shape[0], edge_length, edge_length))
    attributions['x'] = np.apply_along_axis(sum_to_1, 1, dataset.x_test.float().detach().numpy())
    return attributions

# Scale matrix to sum to 1
def sum_to_1(mat):
    return mat / np.sum(mat)

def rand_baseline(data: np.array):
    rands = np.random.uniform(low=-1.0, high=1.0, size=(data.shape))
    return sum_to_1(rands)

def x_baseline(data: np.array):
    return sum_to_1(data)

def get_integrated_gradients_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module, baselines: torch.Tensor) -> torch.tensor:
    return IntegratedGradients(model).attribute(data, target=target, baselines=baselines)

def get_saliency_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return Saliency(model).attribute(data, target=target)

def get_deeplift_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module, baselines: torch.Tensor) -> torch.tensor:
    return DeepLift(model).attribute(data, target=target, baselines=baselines)

def get_deepshap_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module, baselines: torch.Tensor) -> torch.tensor:
    return DeepLiftShap(model).attribute(data.float(), target=target, baselines=baselines)

def get_gradient_shap_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module, baselines: torch.Tensor) -> torch.tensor:
    return GradientShap(model).attribute(data.float(), target=target, baselines=baselines)

def get_guided_backprop_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return GuidedBackprop(model).attribute(data, target=target)

# Hard-coded to target a specific layer for the specific CNN architecture
def get_guided_gradcam_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return GuidedGradCam(model, model.cnn_layers[9]).attribute(data, target=target)

def get_deconvolution_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return Deconvolution(model).attribute(data, target=target)

def get_shapley_sampling_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module, baselines: torch.Tensor) -> torch.tensor:
    return ShapleyValueSampling(model).attribute(data, target=target, baselines=baselines)

def get_lime_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module, baselines: torch.Tensor) -> torch.tensor:
    return Lime(model).attribute(data, target=target, baselines=baselines)

def get_kernel_shap_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module, baselines: torch.Tensor) -> torch.tensor:
    return KernelShap(model).attribute(data, target=target, baselines=baselines)

def get_lrp_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return LRP(model).attribute(data, target=target)

def get_pfi_attributions(data: torch.Tensor, target:torch.Tensor, model: torch.nn.Module) -> torch.tensor:
    return FeaturePermutation(model).attribute(data, target=target)


# https://captum.ai/api/
methods_dict = {
    'Integrated Gradients': get_integrated_gradients_attributions,
    'Saliency': get_saliency_attributions,
    'DeepLift': get_deeplift_attributions,
    'DeepSHAP': get_deepshap_attributions,
    'Gradient SHAP': get_gradient_shap_attributions,
    'Guided Backprop': get_guided_backprop_attributions,
    'Guided GradCAM': get_guided_gradcam_attributions,
    'Deconvolution': get_deconvolution_attributions,
    'Shapley Value Sampling': get_shapley_sampling_attributions,
    'LIME': get_lime_attributions,
    'Kernel SHAP': get_kernel_shap_attributions,
    'LRP': get_lrp_attributions,
    'PFI': get_pfi_attributions
}
