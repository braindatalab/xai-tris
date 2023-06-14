from typing import Dict
from torch import load, manual_seed, device
from torch.cuda import is_available
import numpy as np
from common import XAIScenarios, SEED
from utils import load_json_file,  dump_as_pickle, load_pickle
from training.models import CNN

from xai.methods import get_attributions, get_baselines
import gc

from glob import glob
import sys

np.random.seed(SEED)
manual_seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

DEVICE = device('cuda' if is_available() else 'cpu')

def apply_xai(config: Dict) -> XAIScenarios:
    results = XAIScenarios    
    print('Starting XAI methods')

    data = {}
    data_paths = glob(f'{config["data_path"]}/{sys.argv[1]}*_{sys.argv[2]}*')
    for data_path in data_paths:
        data_scenario = load_pickle(file_path=data_path)
        for key,value in data_scenario.items():
            data[key] = value

    # relatively hard-coded right now as 64x64 results were just on 1 experiment
    experiments_regex = '_0_*'
    if config["num_experiments"] > 1:
        experiments_regex = ''
    torch_model_paths = glob(f'{config["training_output"]}/{sys.argv[1]}*_{sys.argv[2]}*{experiments_regex}.pt')

    for scenario_name, dataset in data.items():
        results[scenario_name] = {
            'LLR': [],
            'MLP': [],
            'CNN': [],
            'scenario': scenario_name
        }
        for model_name in ['LLR', 'MLP', 'CNN']:
            if model_name == 'LLR' and 'linear' not in sys.argv[1]:
                continue
            for model_path in torch_model_paths:
                if model_name in model_path:
                    model = load(model_path).to(DEVICE)
                    results[scenario_name][model_name].append(get_attributions(model=model, dataset=dataset, methods=config['methods'], test_size=config['test_size'], mini_batch=config['mini_batch']))
            
            del model
            gc.collect()

        attributions = get_baselines(dataset, methods=config['filter_methods'], test_size=config['test_size'])
        for key, value in attributions.items():
            results[scenario_name][key] = value

    return results
        

def main():
    print('loading config')
    config = load_json_file(file_path='xai/xai_config.json')
    xai_results = apply_xai(config=config)
    fname = f'{sys.argv[1]}_{sys.argv[2]}_xai_records'
    out_dir = f'{config["output_dir"]}/{sys.argv[3]}'
    dump_as_pickle(data=xai_results, output_dir=out_dir, file_name=fname)


if __name__ == '__main__':
    main()
