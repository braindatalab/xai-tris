import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
plt.style.use('seaborn-colorblind')

# import pandas as pd
import random
import pickle as pkl

from math import floor
import torch
import seaborn as sns
import pandas as pd

from utils import load_json_file
from pathlib import Path

from glob import glob

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
    "Shapley Value Sampling": "Shapley\nSampling",
    "LIME": "LIME",
    "Kernel SHAP": "Kernel\nSHAP",
    "LRP": "LRP",
    "pattern.net": "PatternNet",
    "pattern.attribution": "PatternAttr.",
    "deep_taylor": "DTD",
    "sobel": "Sobel",
    "laplace": "Laplace"
}

abbrev_dict = {
    'linear\nuncorrelated': 'LIN WHITE',
    'linear\ncorrelated': 'LIN CORR',
    'linear\nimagenet': 'LIN IMAGENET',
    'multiplicative\nuncorrelated': 'MULT WHITE',
    'multiplicative\ncorrelated': 'MULT CORR',
    'multiplicative\nimagenet': 'MULT IMAGENET',
    'translations\nrotations\nuncorrelated': 'RIGID WHITE',
    'translations\nrotations\ncorrelated': 'RIGID CORR',
    'translations\nrotations\nimagenet': 'RIGID IMAGENET',
    'xor\nuncorrelated':'XOR WHITE',
    'xor\ncorrelated': 'XOR CORR',
    'xor\nimagenet': 'XOR IMAGENET'
}

def model_plot(boxplot_dict, metric='Precision', predictions='correct', out_dir=None, exp_size=25, palette='colorblind'):
    plt.rcParams['ytick.labelsize'] = 16
    ncol = 1
    bbox_anchor = (-1.65, 7.5)
    figsize = (22,20)
    suptitle = f'{metric} results for explanations produced by samples {predictions}ly predicted'

    x_order=[
        "PFI", "Int. Grad.", "Saliency", "DeepLift", "DeepSHAP", "Grad SHAP", 
        "Deconv", "Shapley Value Sampling",
        "LIME", "Kernel SHAP", "LRP", "PatternNet", "PatternAttr.",
        "DTD", "Guided Backprop.", "Guided GradCAM", "Sobel", "Laplace", "rand","x"
        ]
    
    row_order = [
        'linear\nuncorrelated', 'linear\ncorrelated', 'linear\nimagenet', 'multiplicative\nuncorrelated', 'multiplicative\ncorrelated','multiplicative\nimagenet',
        'translations\nrotations\nuncorrelated','translations\nrotations\ncorrelated','translations\nrotations\nimagenet','xor\nuncorrelated','xor\ncorrelated','xor\nimagenet'
    ]

    col_order = ['LLR', 'MLP', 'CNN']

    dd = pd.DataFrame(boxplot_dict)
    print(dd)
    f = sns.catplot(x="Method", y=metric, hue="Methods", palette=palette, col='model', row='datasets', data=dd, legend_out=True, kind='box', row_order=row_order, col_order=col_order, order=x_order, hue_order=x_order)
    print(f)
    legend_handles = f.legend.legendHandles
    f.fig.figsize = figsize
    plt.close(fig=f.fig)
    
    # if 'Precision' in metric or 'AUROC' in metric:
    #     g = sns.catplot(x="Methods", y=metric, palette=palette, row='datasets', col='model', data=dd, ci='sd', kind="point", row_order=row_order, order=x_order, hue_order=x_order)
    # else:
    g = sns.catplot(x="Methods", y=metric, palette=palette, row='datasets', col='model', data=dd, ci='sd', kind="box", notch=True, row_order=row_order, col_order=col_order, order=x_order, showfliers=False, hue_order=x_order)
    # if model == 'LLR':
    #     g = sns.catplot(x="Method", y=metric, palette='colorblind', col='model', data=dd, ci='sd', kind="point")
    # else:
    # g = sns.catplot(x="Methods", y=metric, palette=palette, col='model', row='datasets', data=dd, ci='sd', kind="point")
    
    g.fig.set_figwidth(figsize[0])
    g.fig.set_figheight(figsize[1])
    
    ax = g.axes
    configure_axes(ax, metric=metric)
    
    axs=ax
    
    plt.legend(handles=legend_handles, bbox_to_anchor=bbox_anchor, prop={'size': 20},
               ncol=ncol, fancybox=True, loc='upper center', borderaxespad=0., facecolor='white', framealpha=0.5)

    plt.subplots_adjust(top=0.92)
    plt.subplots_adjust(wspace=0.05, hspace=0.12)    

    if out_dir:
        plt.savefig(f'{out_dir}/model_full_{metric}_{predictions}_{exp_size}.png', format='png', bbox_inches='tight', dpi=300)


def configure_axes(ax, metric='EMD'):
    print(ax.shape)
    for row_idx in range(ax.shape[0]):
        t = ax[row_idx, 0].get_title(loc='center')
        name_of_metric = t.split('|')[0].split('=')[-1].strip()

        ax[row_idx, 0].set_ylabel(ylabel=abbrev_dict[name_of_metric].replace(' ', '\n'),
                                    fontdict={'fontsize': 20})
        if row_idx > 2:
            ax[row_idx, 0].yaxis.set_label_coords(0.9575,0.5)
        
        for col_idx in range(ax.shape[1]):
            if 0 == row_idx:
                t = ax[row_idx, col_idx].get_title(loc='center')
                new_title = t.split('|')[-1].strip().split('=')[-1].strip()
                ax[row_idx, col_idx].set_title(
                    label=new_title, fontdict={'fontsize': 22})
            else:
                ax[row_idx, col_idx].set_title(label='')
            ax[row_idx, col_idx].set_xlabel(xlabel='',
                                            fontdict={'fontsize': 16})
            labels = ax[row_idx, col_idx].get_xticklabels()
            ax[row_idx, col_idx].set_xticklabels('')
            ax[row_idx, col_idx].patch.set_edgecolor('black')
            
            sns.despine(ax=ax[row_idx, col_idx],
                        top=False, bottom=False, left=False, right=False)
            ax[row_idx, col_idx].grid(True)
            if metric=='EMD':
                ax[row_idx, col_idx].set_ylim(0.6, 1.0)
            if row_idx > 2:
                if col_idx == 0:
                    ax[row_idx, col_idx].xaxis.set_visible(False)
                    ax[row_idx, col_idx].yaxis.grid(False, which='both')
                    plt.setp(ax[row_idx, col_idx].spines.values(), visible=False)
                    ax[row_idx, col_idx].tick_params(left=False, labelleft=False)
                    ax[row_idx, col_idx].patch.set_visible(False)
                else:
                    if col_idx==1:
                        ax[row_idx, col_idx].tick_params(left=True, labelleft=True)
                        if metric=='EMD':
                            ax[row_idx, col_idx].set_yticks([0.62,0.7,0.8,0.9,0.98])
                            ax[row_idx, col_idx].set_yticklabels([0.6,0.7,0.8,0.9,1.0])
                        else:
                            ax[row_idx, col_idx].set_yticks([0.02,0.25,0.5,0.75,0.98])
                            ax[row_idx, col_idx].set_yticklabels([0.0,0.25,0.5,0.75,1.0])
                        

def main():
    print('loading config')
    config = load_json_file(file_path='eval/eval_config.json')
    
    out_dir = f'{config["out_dir"]}/figures'
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    random.seed(2022)
    palette = sns.color_palette('tab20')
    random.shuffle(palette)
    random.shuffle(palette)

    for exp_size in [config['num_experiments']]:
        llr_dict = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'Precision_4': list(), 'EMD_4': list()
        }
        mlp_dict = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'Precision_4': list(), 'EMD_4': list()
        }

        cnn_dict =  {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'Precision_4': list(), 'EMD_4': list()
        }
        all_dicts = {'LLR': llr_dict, 'MLP': mlp_dict, 'CNN': cnn_dict}

        combined_dict = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(),  'model': list(), 'models': list()
        }

        new_dict = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'Precision_4': list(), 'EMD_4': list(), 'model': list(), 'models': list()
        }

        combined_dict_keras = {
            'Method': list(), 'Methods': list(), 'background': list(), 'scenario': list(), 'dataset': list(), 'datasets': list(),
            'Precision': list(), 'EMD': list(), 'model': list(), 'models': list()
        }

        used_keys_comb = combined_dict.keys()

        for i in range(exp_size):
            boxplot_dict = {'results': {}}
            boxplot_dict_keras = {'results': {}}
            torch_paths = glob(f'{config["out_dir"]}/*_quantitative_results.pkl')
            for torch_path in torch_paths:
                with open(torch_path, 'rb') as file:
                    torch_results = pkl.load(file)
                
                # key = scenario name as per data val = results
                for key, val in torch_results['results'].items():
                    boxplot_dict['results'][key] = val
                
            # if also using quantitative results from iNNvestigate methods
            keras_paths = glob(f'{config["out_dir"]}/*_quantitative_results_keras.pkl')
            for keras_path in keras_paths:
                with open(keras_path, 'rb') as file:
                    keras_results = pkl.load(file)
                
                for key, val in keras_results['results'].items():
                    boxplot_dict_keras['results'][key] = val

            for scenario, scenario_results in boxplot_dict['results'].items():
                for model_name, model_results in scenario_results['correct'].items():
                    results_len = 0
                    keras_len = 0
                    for j, result in enumerate(model_results[:5]):
                        for key, value in result.items():
                            if key in used_keys_comb:
                                val = value
                                if model_name == 'CNN':
                                    
                                    if 'xor' not in scenario:
                                        keras_val = boxplot_dict_keras['results'][scenario]['correct'][model_name][j][key]    
                                else:
                                    keras_val = boxplot_dict_keras['results'][scenario]['correct'][model_name][j][key]
                                # keras_val = boxplot_dict_keras['results'][scenario]['correct'][model_name][j][key]
                                if key == 'Method' or key == 'Methods':
                                    val = [m.replace('\n', ' ') for m in value]
                                    if 'xor' in scenario and model_name == 'CNN':
                                        keras_val = []
                                    else:
                                        keras_val = [m.replace('\n', ' ') for m in boxplot_dict_keras['results'][scenario]['correct'][model_name][j][key]]
                                    # val += keras_val
                                combined_dict[key] += val
                                combined_dict_keras[key] += keras_val
                        
                        if model_name == 'CNN':
                            if 'xor' in scenario:
                                keras_len += 0
                            else:
                                keras_len += len(boxplot_dict_keras['results'][scenario]['correct'][model_name][j]['Methods'])        
                        else:
                            keras_len += len(boxplot_dict_keras['results'][scenario]['correct'][model_name][j]['Methods'])
                        results_len += len(result['Methods']) 

                        combined_dict['dataset'] += [m.replace('_','\n')+'\n'+n for m,n in zip(result["scenario"],result["background"])]
                        combined_dict['datasets'] += [m.replace('_','\n')+'\n'+n for m,n in zip(result["scenario"],result["background"])]

                        if 'xor' in scenario and model_name == 'CNN':
                            combined_dict_keras['dataset'] += []
                            combined_dict_keras['datasets'] += []
                        else:
                            combined_dict_keras['dataset'] += [m.replace('_','\n')+'\n'+n for m,n in zip(boxplot_dict_keras['results'][scenario]['correct'][model_name][j]["scenario"],boxplot_dict_keras['results'][scenario]['correct'][model_name][j]["background"])]
                            combined_dict_keras['datasets'] += [m.replace('_','\n')+'\n'+n for m,n in zip(boxplot_dict_keras['results'][scenario]['correct'][model_name][j]["scenario"],boxplot_dict_keras['results'][scenario]['correct'][model_name][j]["background"])]

                        new_dict['dataset'] += [m for m in result["scenario"]]
                        new_dict['datasets'] += [m for m in result["scenario"]]


                    combined_dict['model'] += [model_name]*results_len
                    combined_dict['models'] += [model_name]*results_len

                    combined_dict_keras['model'] += [model_name]*keras_len
                    combined_dict_keras['models'] += [model_name]*keras_len
                        
        for key in combined_dict.keys():
            combined_dict[key] += combined_dict_keras[key]

        # baselines only calculated for one model, need to copy over to other model results to show up in box plots
        for method in ['Sobel', 'Laplace', 'x', 'rand']:
            method_inds_llr = np.where((np.array(combined_dict['Method']) == method) & (np.array(combined_dict['scenario']) == 'linear'))[0]
            method_inds_mlp = np.where(np.array(combined_dict['Method']) == method)[0]

            for model, method_inds in {'LLR': method_inds_llr, 'MLP': method_inds_mlp}.items():
                for key in combined_dict.keys():
                    if 'model' in key:
                        continue

                    duplicate_values = np.array(combined_dict[key])[method_inds].copy()

                    combined_dict[key] += list(duplicate_values)
                combined_dict['model'] += [model]*len(duplicate_values)
                combined_dict['models'] += [model]*len(duplicate_values)

        for metric in ['EMD', 'Precision']:
            model_plot(combined_dict, metric=metric, predictions='correct', out_dir=out_dir, exp_size=exp_size, palette=palette)

if __name__ == '__main__':
    main()