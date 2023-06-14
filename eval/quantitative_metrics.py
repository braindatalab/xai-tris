import numpy as np
import pickle as pkl
from multiprocessing import Pool
from utils import load_json_file, dump_as_pickle, load_pickle
from pathlib import Path

from torch import load

from glob import glob
import sys
import gc
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from eval.eval_utils import calculate_metrics


def main():    
    config = load_json_file(file_path='eval/eval_config.json')

    data_paths = glob(f'{config["data_path"]}/{sys.argv[1]}*_{sys.argv[2]}*')
    for data_path in data_paths:
        data = load_pickle(file_path=data_path)

    scenario = list(data.keys())[0]

    # relatively hard-coded right now as 64x64 results were just on 1 experiment
    experiments_regex = '_0_*'
    if config["num_experiments"] > 1:
        experiments_regex = ''
    torch_model_paths = glob(f'{config["training_path"]}/{sys.argv[1]}*_{sys.argv[2]}*{experiments_regex}.pt')

    with open(f'{config["xai_path"]}/{sys.argv[1]}_{sys.argv[2]}_xai_records.pkl', 'rb') as file:
        xai_output = pkl.load(file)

    inds_list = []
    args_list = []

    test_size = config["test_size"]

    for model_name in ['LLR', 'MLP', 'CNN']:
        if model_name == 'LLR' and 'linear' not in sys.argv[1]:
            continue

        model_ind = 0
        for model_path in torch_model_paths:
            if model_name in model_path:
                model = load(model_path, map_location=torch.device('cpu'))
                test_data_torch = data[scenario].x_test
                if model_name == 'CNN':
                    test_data_torch = test_data_torch.reshape(test_data_torch.shape[0], 1, 64, 64)
                torch_otpt = model(test_data_torch.float())
                torch_inds = data[scenario].y_test.detach().numpy() == np.argmax(torch_otpt.detach().numpy(), axis=1)
                inds_list.append(torch_inds)
                del model
                gc.collect()

                args_list.append([xai_output[scenario], scenario, data[scenario].masks_test[:test_size], test_size, model_name, model_ind, config["mini_batch"]])
                model_ind +=1
                
    intersection = np.logical_and.reduce(inds_list)
    print(intersection)

    for i in range(len(args_list)):
        args_list[i].append(intersection)
    
    pool = Pool(processes=config["num_threads"])
    results = pool.starmap_async(calculate_metrics, args_list)

    boxplot_dict = {}

    for result in results.get():
        for key, value in result.items():
            if key not in boxplot_dict:
                boxplot_dict[key] = {'correct': {}, 'incorrect': {}}
            else:
                for model_key, model_val in value['correct'].items():
                    if model_key not in boxplot_dict[key]['correct']:
                        boxplot_dict[key]['correct'][model_key] = model_val
                    else:
                        boxplot_dict[key]['correct'][model_key] += model_val
                for model_key, model_val in value['incorrect'].items():
                    if model_key not in boxplot_dict[key]['incorrect']:
                        boxplot_dict[key]['incorrect'][model_key] = model_val
                    else:
                        boxplot_dict[key]['incorrect'][model_key] += model_val

    quantitative_result = {'results': boxplot_dict, 'intersections': intersection}

    out_folder_path = f'{config["out_dir"]}'
    Path(f'{out_folder_path}').mkdir(parents=True, exist_ok=True)

    dump_as_pickle(data=quantitative_result, output_dir=out_folder_path, file_name=f'{sys.argv[1]}_{sys.argv[2]}_quantitative_results')

if __name__ == '__main__':
    main()
