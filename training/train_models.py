from typing import List, Dict, Tuple
import torch

from common import TrainingRecord, TrainingScenarios, SEED
from utils import load_json_file, append_date, dump_as_pickle, load_pickle

import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset

from torchmetrics import Accuracy
from tqdm import tqdm
from pathlib import Path

import gc
import sys

import pandas as pd
import numpy as np
np.random.seed(SEED)

from torch import manual_seed
manual_seed(SEED)

import os
os.environ['PYTHONHASHSEED']=str(SEED)

import random
random.seed(SEED)

from glob import glob

# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')
os.environ["CUDA_VISIBLE_DEVICES"]=""

from training.models import models_dict, init_he_normal

print(torch.cuda.is_available())

# def l1_penalty(model, lambda_l1=0.0001):
#     lossl1 = 0
#     for model_param_name, model_param_value in model.named_parameters():
#         if model_param_name.endswith('weight'):
#             lossl1 += lambda_l1 * model_param_value.abs().sum()
#     return lossl1

def train(config: Dict, data: List, lr: float, folder_path: str, experiment_num: int, scenarios_checkpoint: TrainingScenarios=None, restart_point: Tuple=None):
    n_dim = data[list(data.keys())[0]].x_test.shape[1]
    edge_length = int(np.sqrt(n_dim))

    scenarios = TrainingScenarios
    start_from_checkpoint = False
    if scenarios_checkpoint != None:
        scenarios = scenarios_checkpoint
        start_from_checkpoint = True
        restarted = False
    scenarios['data_path'] = config['data_path']

    models = ['LLR', 'MLP', 'CNN']
    accs = {'val': pd.Series(), 'test': pd.Series()}

    
    for i, (data_params, data_record) in enumerate(data.items()):
        scenarios['training_records'][data_params] = pd.Series()
        accs['val'][data_params] = pd.Series()
        accs['test'][data_params] = pd.Series()

        #kwargs = {'num_workers':1, 'pin_memory': True} if torch.cuda.is_available() else {}
        kwargs = {'pin_memory': False}
        train_loader = DataLoader(TensorDataset(data_record.x_train, data_record.y_train),
                batch_size=config['batch_size'], shuffle=True, **kwargs)
                    
        val_loader = DataLoader(TensorDataset(data_record.x_val, data_record.y_val), 
                batch_size=config['batch_size'], shuffle=False, **kwargs)

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
            records = []
            for j in range(config['runs_per_model']):
                if restart_point != None and model_name == restart_point[0] and restarted == False:
                    if j < restart_point[1]:
                        continue
                    else:
                        restarted = True
                    print(f'Restarting training for {data_params} experiment {experiment_num} at {model_name} run {j+1}')
                
                # Early Stopping variables
                min_loss = 10000000
                trigger_times = 0

                print(f'\nTraining Parameterization {data_params} with model {model_name} (experiment {experiment_num}, {j+1}/{config["runs_per_model"]})')

                # fully connected layer size for CNN
                if n_dim == 64:
                    linear_dim = 4
                elif n_dim == 64*64:
                    linear_dim = 288
                elif n_dim == 128*128:
                    linear_dim = 256

                if 'CNN' in model_name:
                    model = models_dict.get(model_name)(n_dim, linear_dim)
                    save_model = type(model)(n_dim, linear_dim)
                else:
                    model = models_dict.get(model_name)(n_dim)
                    save_model = type(model)(n_dim)
                
                # He normal weight initialization
                model.apply(init_he_normal)

                model = model.to(DEVICE)
                save_model = save_model.to(DEVICE)

                # defining the loss function
                criterion = CrossEntropyLoss()
                criterion = criterion.to(DEVICE)

                # defining the optimizer
                optimizer = Adam(model.parameters(), lr=lr)
                #optimizer = Adam(model.parameters(), lr=lr, weight_decay=0.05)

                training_accuracy = Accuracy(task='binary').to(DEVICE)
                validation_accuracy = Accuracy(task='binary').to(DEVICE)
                test_accuracy = Accuracy(task='binary').to(DEVICE)

                training_loss = np.array([])
                validation_loss = np.array([])

                training_accuracies = np.array([])
                validation_accuracies = np.array([])
                for epoch in tqdm(range(config['epochs'])):
                    epoch_accuracies = np.array([])
                    epoch_validation_accuracies = np.array([])
                    epoch_loss = 0
                    epoch_validation_loss = 0
                    num_batches_train = 0
                    for x, y in train_loader:
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)

                        if 'CNN' in model_name:
                            x = x.reshape(x.shape[0], 1, edge_length, edge_length)
                        optimizer.zero_grad()
                        
                        output = model(x.float())
                        loss = criterion(output, y)
                        #loss = criterion(output, y) + l1_penalty(model, 0.00001)
                        loss.backward()
                        optimizer.step()
                        epoch_loss += loss.item()
                        num_batches_train += 1
                        acc = training_accuracy(torch.argmax(output, dim=1), y).item()
                        epoch_accuracies = np.append(epoch_accuracies, acc)
                        del x
                        del y
                        del acc
                        del loss 

                    training_accuracies = np.append(training_accuracies, sum(epoch_accuracies)/len(epoch_accuracies))

                    num_batches_validation = 0
                    save_acc = 0
                    for x, y in val_loader:
                        x = x.to(DEVICE)
                        y = y.to(DEVICE)
                        
                        if 'CNN' in model_name:
                            x = x.reshape(x.shape[0], 1, edge_length, edge_length)
                        output = model(x.float())
                        loss = criterion(output, y)

                        epoch_validation_loss += loss.item()
                        acc = validation_accuracy(torch.argmax(output, dim=1), y).item()
                        save_acc += acc
                        num_batches_validation += 1
                        
                        epoch_validation_accuracies = np.append(epoch_validation_accuracies, acc)
                        del x
                        del y
                        del acc
                        del loss 
                    
                    if epoch_validation_loss <= min_loss:
                        min_loss = epoch_validation_loss
                        # get a new instance
                        if 'CNN' in model_name:
                            save_model = type(model)(n_dim, linear_dim)
                        else:
                            save_model = type(model)(n_dim)
                        save_model.load_state_dict(model.state_dict()) # copy weights and stuff
                        save_acc = save_acc/num_batches_validation
                    
                    #if (sum(epoch_validation_accuracies)/len(epoch_validation_accuracies) >= 0.99):
                    #    print("Early stopping due to val acc breaching 99% threshold!")
                    #    break
                    validation_accuracies = np.append(validation_accuracies, sum(epoch_validation_accuracies)/len(epoch_validation_accuracies))

                    epoch_loss /= num_batches_train
                    epoch_validation_loss /= num_batches_validation

                    training_loss = np.append(training_loss, epoch_loss)
                    validation_loss = np.append(validation_loss, epoch_validation_loss)

                    if epoch_validation_loss > min_loss:
                        trigger_times += 1
                        if trigger_times >= config['patience']:
                            print("Early stopping due to exceeded patience threshold!")
                            break
                    else:
                        trigger_times = 0
                    gc.collect()
                del model
                del optimizer
                del criterion
                test_data = data_record.x_test
                if 'CNN' in model_name:
                    test_data = data_record.x_test.reshape(data_record.x_test.shape[0], 1, edge_length, edge_length)

                otpt = save_model(test_data.float())

                torch_inds = data_record.y_test.detach().numpy() == np.argmax(otpt.detach().numpy(), axis=1)
                test_acc = test_accuracy(torch.argmax(otpt, dim=1), data_record.y_test).item()
                test_accs = np.append(test_accs, test_acc)
                avg_test_acc += test_acc
                avg_acc += validation_accuracies[-1]
                
                acc_string = f'Test Accuracy for scenario {data_params} with model {model_name}: {test_acc} (experiment {experiment_num}, {j+1}/{config["runs_per_model"]})'
                print(acc_string)

                del test_data
                del test_acc
                save_path = f'{folder_path}/{data_params}_{model_name}_{experiment_num}_{j}.pt'
                torch.save(save_model, save_path)        

                train_record = TrainingRecord(f'{data_params}_{model_name}_{j}',
                                            save_path, save_path, training_loss, validation_loss, 
                                            training_accuracies, validation_accuracies, torch_inds)

                records.append(train_record)
                
                del save_model
                
                gc.collect()
            scenarios['training_records'][data_params][model_name]['records'] = records

            acc_string = f'Avg Accuracy for scenario {data_params} with model {model_name} experiment {experiment_num}, lr {lr}: {avg_acc / config["runs_per_model"]}'
            
            print(acc_string)
            
            accs['val'][data_params][model_name] = avg_acc / config['runs_per_model']
            accs['test'][data_params][model_name] = avg_test_acc / config['runs_per_model']
            print(accs['test'][data_params])
            scenarios['training_records'][data_params][model_name]['avg_acc'] = avg_acc / config['runs_per_model']
            scenarios['training_records'][data_params][model_name]['avg_test_acc'] = avg_test_acc / config['runs_per_model']
            scenarios['training_records'][data_params][model_name]['test_accs'] = test_accs

            fname = append_date(s=f'{config["data_scenario"]}_training_records')
            dump_as_pickle(data=scenarios, output_dir=folder_path, file_name=fname)
        del train_loader
        del val_loader
    
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

    scenarios_checkpoint = None
    restart_point = None

    lr = 0.0005

    scenarios = train(config=config, data=data_scenario, lr=lr, folder_path=folder_path, experiment_num=int(sys.argv[3]), scenarios_checkpoint=scenarios_checkpoint, restart_point=restart_point)

    dump_as_pickle(data=scenarios, output_dir=out_folder_path, file_name=f'{sys.argv[1]}_{sys.argv[2]}_log')


if __name__ == '__main__':
    main()
