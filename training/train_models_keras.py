import re
from typing import List, Dict, Tuple
import json
# import torch

from datetime import datetime
from common import TrainingRecord, TrainingScenarios, SEED
from utils import load_json_file, append_date, dump_as_pickle, load_pickle

from tqdm import tqdm
from pathlib import Path

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from training.models_keras import models_dict

import pandas as pd

import sys

from tensorflow import set_random_seed
set_random_seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

import numpy as np
np.random.seed(SEED)

from math import floor

from glob import glob

import tensorflow as tf
from keras import backend as K

# class AccuracyStopping(Callback):
#     def __init__(self):
#         super(AccuracyStopping, self).__init__()

#     def on_epoch_end(self, batch, logs={}):
#         val_acc = logs.get('val_acc')
#         if val_acc > 0.99:
#             self.model.stop_training = True


def train(config: Dict, data: List, lr: float, folder_path: str, experiment_num: int, scenarios_checkpoint: TrainingScenarios=None, restart_point: Tuple=None):
    scenarios = TrainingScenarios

    start_from_checkpoint = False
    if scenarios_checkpoint != None:
        scenarios = scenarios_checkpoint
        start_from_checkpoint = True
        restarted = False

    scenarios['data_path'] = config['data_path']
    accs = {'val': pd.Series(), 'test': pd.Series()}

    models = ['LLR_Keras','MLP_Keras','CNN_Keras']

    for i, (data_params, data_record) in enumerate(data.items()):
        scenarios['training_records'][data_params] = {}
        accs['val'][data_params] = pd.Series()
        accs['test'][data_params] = pd.Series()

        for model_name in models:
            if 'linear' not in data_params and 'LLR' in model_name:
                    continue
            if restart_point != None and start_from_checkpoint == True:
                if model_name != restart_point[0]:
                    continue
                else:
                    start_from_checkpoint = False

            if model_name not in scenarios['training_records'][data_params].keys():
                scenarios['training_records'][data_params][model_name] = {
                    'records': [],
                    'avg_acc': 0,
                    'avg_test_acc': 0,
                    'test_accs': np.array([])
                }
                avg_acc = 0
                avg_test_acc = 0
                test_accs = np.array([])
            else:
                avg_acc = scenarios['training_records'][data_params][model_name]['avg_acc']
                avg_test_acc = scenarios['training_records'][data_params][model_name]['avg_test_acc']
                test_accs = scenarios['training_records'][data_params][model_name]['test_accs']

            for j in range(config['runs_per_model']):
                if restart_point != None and model_name == restart_point[0] and restarted == False:
                    if j < restart_point[1]:
                        continue
                    else:
                        restarted = True
                    print(f'Restarting training for {data_params} at {model_name} run {j+1}')
                print(f'\nTraining Parameterization {data_params} with model {model_name}, lr {lr} (experiment {experiment_num} {j+1}/{config["runs_per_model"]})')
                train_data = data_record.x_train.detach().numpy()

                train_labels = data_record.y_train.detach().numpy()
                train_labels = to_categorical(train_labels.astype('float32'), num_classes=2)
            
                val_data = data_record.x_val.detach().numpy()
                val_labels = data_record.y_val.detach().numpy()
                val_labels = to_categorical(val_labels.astype('float32'), num_classes=2)

                test_data = data_record.x_test.detach().numpy()
                test_labels = data_record.y_test.detach().numpy()
                test_labels = to_categorical(test_labels.astype('float32'), num_classes=2)


                if 'CNN' in model_name:
                    train_data = train_data.reshape(train_data.shape[0], 64, 64, 1)
                    val_data = val_data.reshape(val_data.shape[0], 64, 64, 1)
                    test_data = test_data.reshape((test_data.shape[0], 64, 64, 1))

                model = models_dict.get(model_name)(lr)

                save_path = f'{folder_path}/{data_params}_{model_name}_{experiment_num}_{j}.h5'

                model_checkpoint_callback = ModelCheckpoint(
                        filepath=save_path,
                        save_weights_only=False,
                        verbose=0,
                        monitor='val_loss',
                        mode='min',
                        save_best_only=True)
                # acc_callback = AccuracyStopping()
                my_callbacks = [EarlyStopping(patience=config['patience']), model_checkpoint_callback]
                history = model.fit(train_data, train_labels, epochs=config['epochs'], validation_data=(val_data, val_labels),
                                    verbose=True, batch_size=config['batch_size'], callbacks=my_callbacks)
                training_accuracies = history.history['acc']
                validation_accuracies = history.history['val_acc']
                training_loss = history.history['loss']
                validation_loss = history.history['val_loss']

                otpt = model.predict(test_data)

                keras_inds = np.argmax(test_labels, axis=1) == np.argmax(otpt, axis=1)
                test_acc = np.sum(keras_inds==True)/int(test_data.shape[0])
                test_accs = np.append(test_accs, test_acc)
                avg_acc += validation_accuracies[-1]
                avg_test_acc += test_acc

                train_record = TrainingRecord(f'{data_params}_{model_name}_{j}',
                                        save_path, save_path, training_loss, validation_loss, 
                                        training_accuracies, validation_accuracies, keras_inds)
                scenarios['training_records'][data_params][model_name]['records'] += [train_record]

                acc_string = f'Test Accuracy for scenario {data_params} with model {model_name} experiment {experiment_num}, lr {lr}: {test_acc} ({j+1}/{config["runs_per_model"]})'
                print(acc_string)
            acc_string = f'Avg Test Accuracy for scenario {data_params} with model {model_name} experiment {experiment_num}, lr {lr}: {avg_test_acc / config["runs_per_model"]}'
            print(acc_string)

            scenarios['training_records'][data_params][model_name]['avg_acc'] = avg_acc / config['runs_per_model']
            scenarios['training_records'][data_params][model_name]['avg_test_acc'] = avg_test_acc / config['runs_per_model']
            scenarios['training_records'][data_params][model_name]['test_accs'] = test_accs

            fname = append_date(s=f'{config["data_scenario"]}_training_records_keras')
            dump_as_pickle(data=scenarios, output_dir=folder_path, file_name=fname)

    return scenarios


def main():
    config = load_json_file(file_path='training/training_config.json')


    folder_path = f'{config["output_dir"]}/{sys.argv[4]}'
    out_folder_path = f'{folder_path}/logs/{sys.argv[3]}'
    Path(f'{out_folder_path}').mkdir(parents=True, exist_ok=True)

    data_paths = glob(f'{config["data_path"]}/{sys.argv[1]}*_{sys.argv[2]}*')
    data_scenario = pd.Series()
    print(data_paths)
    for data_path in data_paths:
        data = load_pickle(file_path=data_path)
        
        for key,value in data.items():
            data_scenario[key] = value
            print(data_scenario[key].x_train.shape)
                        
    if len(data_scenario) == 0:
        raise Exception(f'Scenario {sys.argv[1]} not recognised! Check the specified data_path in training_config.json and try again')

    lr = 0.0005

    scenarios_checkpoint = None
    restart_point = None

    scenarios = train(config=config, data=data_scenario, lr=lr, folder_path=folder_path, experiment_num=int(sys.argv[3]), scenarios_checkpoint=scenarios_checkpoint, restart_point=restart_point)

    dump_as_pickle(data=scenarios, output_dir=out_folder_path, file_name=f'{sys.argv[1]}_{sys.argv[2]}_log_keras')



if __name__ == '__main__':
    main()
