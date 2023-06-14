import numpy as np

from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter

from ot.lp import emd

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

NAME_MAPPING = {
    "PFI": "PFI",
    "Integrated Gradients": "Int. Grad.",
    "Saliency": "Saliency",
    "DeepLift": "DeepLift",
    "DeepSHAP": "DeepSHAP",
    "Gradient SHAP": "Grad\nSHAP",
    "Guided Backprop": "Guided\nBackprop.",
    "Guided GradCAM": "Guided\nGradCAM",
    "Deconvolution": "Deconv",
    "Shapley Value Sampling": "Shapley\nValue\nSampling",
    "LIME": "LIME",
    "Kernel SHAP": "Kernel\nSHAP",
    "LRP": "LRP",
    "pattern.net": "PatternNet",
    "pattern.attribution": "PatternAttr.",
    "deep_taylor": "DTD",
    "sobel": "Sobel",
    "laplace": "Laplace"
}

# n_dim = 64**2
normal_t = [[1,0],[1,1],[1,0]]
normal_l = [[1,0],[1,0],[1,1]]
combined_mask = np.zeros((8,8))
combined_mask[1:4, 1:3] = normal_t
combined_mask[4:7, 5:7] = normal_l
combined_mask = combined_mask.reshape((8,8))

combined_mask_pat_scaled = np.kron(combined_mask, np.ones((8,8)))
combined_mask_smoothed = gaussian_filter(combined_mask_pat_scaled, 1.5)
combined_mask_binary = combined_mask_smoothed.copy()
combined_mask_binary[combined_mask_smoothed>=0.05] = 1
combined_mask_binary[combined_mask_smoothed<0.05] = 0

def create_cost_matrix(edge_length=64):
    mat = np.indices((edge_length,edge_length))
    coords = []
    for i in range(edge_length):
        for j in range(edge_length):
            coords.append((mat[0][i][j], mat[1][i][j]))
    coords = np.array(coords)
    return cdist(coords,coords)

cost_matrix_64by64 = create_cost_matrix(64)
cost_matrix_8by8 = create_cost_matrix(8)

# Scale matrix to sum to 1
def sum_to_1(mat):
    return mat / np.sum(mat)


# Calculate EMD for full, continuous-valued, attribution
# score = 1-(EMD/Dmax), where Dmax = max euclidean distance, aka sqrt(7^2 + 7^2)=9.8995 for the 8x8 data
def continuous_emd(gt_mask, attribution, n_dim=64):
    cost_matrix = cost_matrix_64by64
    if n_dim == 64:
        cost_matrix = cost_matrix_8by8

    _, log = emd(sum_to_1(gt_mask.reshape(n_dim)).astype(np.float64), sum_to_1(np.abs(attribution).reshape(n_dim)).astype(np.float64), cost_matrix, numItermax=200000, log=True)

    return 1 - (log['cost']/np.sqrt(n_dim + n_dim))

def precision(gt_mask, attribution, n_dim=64, n=8):
    # get highest n non-zero attribution values 
    non_zero_inds = np.where(attribution!=0)[0]
    sort_n=min(n, len(non_zero_inds))
    ordered = np.argpartition(np.abs(attribution[non_zero_inds]), -sort_n)[-sort_n:]
    
    # assign highed inds to a 'mask' for comparison with ground truth
    ind = non_zero_inds[ordered]
    data_mask = np.zeros((n_dim))
    data_mask[ind] = 1
    
    inds_gt = np.where(gt_mask == 1)[0]
    inds_attr = np.where(data_mask == 1)[0]
    overlapping = np.intersect1d(inds_gt, inds_attr) # precision = overlap between n most important pixels and n ground truth pixels
    return len(overlapping)/n, data_mask

def top_8_score(combined_mask, attributions, n_dim=64):
    length = attributions.shape[0]
    edge_length = int(np.sqrt(n_dim))
    scores = []
    data_masks = []
    continuous_emd_scores = []
    n = len(np.where(combined_mask.reshape(n_dim) != 0)[0])
    for i in range(length):
        try:
            prec_score, data_mask = precision(combined_mask.reshape(n_dim), attributions[i].reshape(n_dim), n_dim=n_dim, n=n)

            normalised_gt = sum_to_1(combined_mask.reshape(n_dim))
            normalised_attribution = sum_to_1(np.abs(attributions[i]).reshape(n_dim))
            continuous_emd_score = continuous_emd(normalised_gt.reshape((edge_length,edge_length)), normalised_attribution.reshape((edge_length,edge_length)), n_dim=n_dim)

            scores.append(prec_score)
            data_masks.append(data_mask)
            continuous_emd_scores.append(continuous_emd_score)
        except:
            continue
        
    return scores, data_masks, continuous_emd_scores

def top_4_score(ground_truths, attributions, n_dim=64):
    length = range(min(ground_truths.shape[0], attributions.shape[0]))
    edge_length = int(np.sqrt(n_dim))
    scores = []
    data_masks = []
    continuous_emd_scores = []

    for i in length:
        try:
            n = len(np.where(ground_truths[i].reshape(n_dim) != 0)[0])
            prec_score, data_mask = precision(ground_truths[i].reshape(n_dim), attributions[i].reshape(n_dim), n_dim=n_dim, n=n)

            normalised_gt = sum_to_1(ground_truths[i].reshape(n_dim))
            normalised_attribution = sum_to_1(np.abs(attributions[i]).reshape(n_dim))
            continuous_emd_score = continuous_emd(normalised_gt.reshape((edge_length,edge_length)), normalised_attribution.reshape((edge_length,edge_length)), n_dim=n_dim)

            scores.append(prec_score)
            data_masks.append(data_mask)
            continuous_emd_scores.append(continuous_emd_score)
        except:
            continue
    return scores, data_masks, continuous_emd_scores

def calculate_metrics(xai_output, scenario, masks_test, test_size, model_name, model_ind, mini_batch, n_dim, intersection):    
    boxplot_dicts = {scenario: {'correct': {}, 'incorrect': {}}}
    for model in [model_name]:
        if 'linear' not in scenario and model == 'LLR':
            continue
        boxplot_dicts[scenario]['correct'][model] = []
        boxplot_dicts[scenario]['incorrect'][model] = []
        for i in [model_ind]:  # len(xai_output[model]) == 5
            correct_dict = {
                'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(),
                'Precision': list(), 'EMD': list(),
                'Precision_4': list(), 'EMD_4': list()
            }
            incorrect_dict = {
                'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(),
                'Precision': list(), 'EMD': list(),
                'Precision_4': list(), 'EMD_4': list()
            }
            
            if i == 0 and 'sobel' in xai_output.keys():
                xai_output[model][i]['sobel'] = xai_output['sobel']
                xai_output[model][i]['laplace'] = xai_output['laplace']
                xai_output[model][i]['rand'] = xai_output['rand']
                xai_output[model][i]['x'] = xai_output['x']
            
            print("calculating metrics for", scenario, model, i)

            # calculate metrics for xai methods
            for method_name, attributions in xai_output[model][i].items():
                correct_inds = intersection
                incorrect_inds = np.invert(correct_inds)
                
                correct_attributions = np.array(attributions)[:test_size].reshape((test_size, n_dim))[correct_inds[:test_size]]
                correct_masks = masks_test[correct_inds[:test_size]]

                incorrect_attributions = np.array(attributions)[:test_size].reshape((test_size, n_dim))[incorrect_inds[:test_size]]
                incorrect_masks = masks_test[incorrect_inds[:test_size]]

                if n_dim == 4096:
                    correct_masks[correct_masks>=0.05] = 1
                    correct_masks[correct_masks<0.05] = 0

                    incorrect_masks[incorrect_masks>=0.05] = 1
                    incorrect_masks[incorrect_masks<0.05] = 0

                    top_8_mask = combined_mask_binary
                else:
                    top_8_mask = combined_mask

                for ind in range(0, test_size, mini_batch):
                    correct_batch = correct_attributions[ind:ind+mini_batch]
                    incorrect_batch = incorrect_attributions[ind:ind+mini_batch]

                    precision_4, _, continuous_emd_scores_4 = top_4_score(correct_masks[ind:ind+mini_batch], correct_batch, n_dim)
                    precision_incorrect_4, _,  continuous_emd_scores_incorrect_4 = top_4_score(incorrect_masks[ind:ind+mini_batch], incorrect_batch, n_dim)
                    
                    if 'translations' in scenario:
                        precision = precision_4
                        continuous_emd_scores = continuous_emd_scores_4

                        precision_incorrect = precision_incorrect_4
                        continuous_emd_scores_incorrect = continuous_emd_scores_incorrect_4

                    else:
                        precision, _, continuous_emd_scores = top_8_score(top_8_mask, correct_batch, n_dim)
                        precision_incorrect, _, continuous_emd_scores_incorrect = top_8_score(top_8_mask, incorrect_batch, n_dim)

                    correct_dict['Precision'] += precision
                    correct_dict['EMD'] += continuous_emd_scores

                    incorrect_dict['Precision'] += precision_incorrect
                    incorrect_dict['EMD'] += continuous_emd_scores_incorrect

                    correct_dict['Precision_4'] += precision_4
                    correct_dict['EMD_4'] += continuous_emd_scores_4

                    incorrect_dict['Precision_4'] += precision_incorrect_4
                    incorrect_dict['EMD_4'] += continuous_emd_scores_incorrect_4

                    mapped = NAME_MAPPING.get(method_name, method_name)
                    correct_dict['Method'] += [mapped] * len(precision)
                    correct_dict['Methods'] += [mapped] * len(precision)
                    
                    if 'translations' in scenario:
                        scen_save = 'translations_rotations'
                    else:
                        scen_save = scenario.split('_')[0]
                    correct_dict['scenario'] += [scen_save] * len(precision)
                    correct_dict['background'] += [scenario.split('_')[-1]] * len(precision)

                    incorrect_dict['Method'] += [mapped] * len(precision_incorrect)
                    incorrect_dict['Methods'] += [mapped] * len(precision_incorrect)
                    
                    incorrect_dict['scenario'] += [scen_save] * len(precision_incorrect)
                    incorrect_dict['background'] += [scenario.split('_')[-1]] * len(precision_incorrect)

            boxplot_dicts[scenario]['correct'][model] += [correct_dict]
            boxplot_dicts[scenario]['incorrect'][model] += [incorrect_dict]

    return boxplot_dicts