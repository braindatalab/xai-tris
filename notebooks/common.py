from collections import namedtuple

SEED = 4000002

DataRecord = namedtuple('DataRecord', 'x_train y_train x_val y_val x_test y_test masks_train masks_val masks_test')

TrainingRecord = namedtuple('TrainingRecord', 'data_params model_path keras_path train_loss val_loss train_accuracy val_accuracy test_preds')

# Usages of DataScenarios: 
# data = DataScenarios
# data[data_params] = DataRecords
DataScenarios = dict()
TrainingScenarios = {'training_records': dict(), 'data_path': str}

# Usages of XAIScenarios: 
# xai = XAIScenarios
# xai[data_params] = attributions 
XAIScenarios = dict()