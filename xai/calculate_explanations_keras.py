from typing import Dict

from keras.models import load_model
from keras.backend import clear_session
import numpy as np
from common import XAIScenarios, SEED
from utils import load_json_file, append_date, dump_as_pickle, load_pickle

from xai.methods_keras import get_keras_attributions
import gc

from glob import glob
import sys

np.random.seed(SEED)

from torch import manual_seed
manual_seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

def apply_xai(config: Dict) -> XAIScenarios:
    results = XAIScenarios    
    print('Starting XAI methods')
    data = {}
    data_paths = glob(f'{config["data_path"]}/{sys.argv[1]}*_{sys.argv[2]}*')
    for data_path in data_paths:
        data_scenario = load_pickle(file_path=data_path)
        for key,value in data_scenario.items():
            data[key] = value

    keras_model_paths = glob(f'./artifacts/tetris/training/{config["training_output"]}/{sys.argv[1]}*_{sys.argv[2]}*_Keras_0_*.h5')

    print(keras_model_paths)
    print(data.keys())
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

            for model_path in keras_model_paths:
                if model_name in model_path:
                    keras_model = load_model(model_path)
                    attributions = get_keras_attributions(model=keras_model, dataset=dataset, methods = config['keras_methods'], test_size=config['test_size'],  mini_batch=config['mini_batch'])
                    results[scenario_name][model_name].append(attributions)

                    clear_session()
                    del keras_model
                    gc.collect()

    return results
        
def main():
    print('loading config')
    config = load_json_file(file_path='xai/xai_config.json')
    xai_results = apply_xai(config=config)
    fname = f'{sys.argv[1]}_{sys.argv[2]}_xai_records_keras'
    out_dir = f'{config["output_dir"]}/{sys.argv[3]}'
    dump_as_pickle(data=xai_results, output_dir=out_dir, file_name=fname)


if __name__ == '__main__':
    main()
