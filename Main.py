import copy
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from model import Net
from smiles_featurizers import morgan_fingerprints, mac_keys_fingerprints, one_hot_encode, morgan_fingerprints_mac_and_one_hot, morgan_fingerprints_and_one_hot
from utils import get_data_dictionaries, create_test_metrics, test_model, get_training_and_test_data, save_dict, plot_history, predictions_scatter_plot, save_model, calculate_metrics, get_smiles_dict, predictions_heat_map



path_to_all_smiles = "/Users/abdulsalamyazid/Desktop/thesis/Projects/Predicting Docking Scores/Data Set/Target1/all_smiles.txt"
PATH_TO_CSV = "/Users/abdulsalamyazid/Desktop/thesis/Projects/Predicting Docking Scores/Data Set/Target1/all_smiles_clean_protein1.csv"
# main dataset
# Program Variables
DATA = pd.read_csv(PATH_TO_CSV)
data_set_size = len(DATA)
number_of_folds = 5
identifier = str(number_of_folds) + " Fold Cross Validation"
TRAINING_SIZE = 1000
if TRAINING_SIZE > 99999:
    TESTING_SIZE = data_set_size - (TRAINING_SIZE * number_of_folds)
else:
    TESTING_SIZE = 3000
TRAINING_SIZE = TRAINING_SIZE * number_of_folds

feature_dim_3_features = 3531
feature_dim_2_features = 3364
feature_dim_one_hot_features = 2340
feature_dim_morgan_features = 1024
feature_dim_mac_features = 167
feature_dim = feature_dim_one_hot_features
int2char, char2int, dict_size = get_smiles_dict(path_to_all_smiles)


class Training(Dataset):

    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict.iloc[idx]
        smile = data['smile']
        score = data['docking_score']
        features = one_hot_encode(smile, char2int, dict_size)
        features = torch.from_numpy(features.reshape(features.shape[0], 1))
        score = torch.tensor([score])
        return features, score


train, test = get_training_and_test_data(DATA, TRAINING_SIZE, TESTING_SIZE)

print("training size ", TRAINING_SIZE)
print("testing size ", TESTING_SIZE)

# Testing the dataloader
smiles_data_train = Training(train)
smiles_data_test = Training(test)

for i in range(len(smiles_data_train)):
    features, score = smiles_data_train[i]

    print(i, features.shape, score)

    if i == 3:
        break

test_dataloader = DataLoader(smiles_data_test, batch_size=128, shuffle=False, num_workers=7)


def train_model(train_dataloader, model, criterion, optimizer, number_of_epochs):
    metrics_dict = {"training_mse": []}
    mse_array_train = []

    for epoch in range(number_of_epochs):
        running_loss = 0
        for i_batch, data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            features, target = data
            features = features.squeeze()
            outputs = net(features).double()
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            del features, outputs
            running_loss += loss.item()
        mse_array_train.append(running_loss / len(train_dataloader))
        print("Epoch: {}/{}.. ".format(epoch + 1, number_of_epochs),
              "Training MSE: {:.3f}.. ".format(running_loss / len(train_dataloader)))

    metrics_dict["training_mse"] = mse_array_train
    return metrics_dict
    print("Finished training.")


start_time_train_val = time.time()
all_train_metrics = []
df_split = np.array_split(train, number_of_folds)
all_networks = []
all_fold_test_mse = []
fold_mse, fold_mae, fold_rquared = 0, 0, 0

for fold in range(number_of_folds):
    print("fold ", fold)
    print()
    number_of_epochs = 6
    net = Net(feature_dim)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    temp_data = copy.deepcopy(df_split)
    temp_data.pop(fold)
    temp_data = pd.concat(temp_data)
    smiles_data_train = Training(temp_data)  # train
    smiles_data_train = Training(train)  # train
    train_dataloader = DataLoader(smiles_data_train, batch_size=128, shuffle=True, num_workers=7)
    criterion = nn.MSELoss()
    # training
    metrics_dict = train_model(train_dataloader, net, criterion,
                               optimizer, number_of_epochs)
    all_networks.append(net)
    all_train_metrics.append(metrics_dict)

    # test
    fold_test_dataloader_class = Training(df_split[fold])
    fold_test_dataloader = DataLoader(fold_test_dataloader_class, batch_size=128, shuffle=False, num_workers=7)
    fold_predictions = test_model(fold_test_dataloader, net)
    test_smiles_target = df_split[fold]['docking_score'].tolist()
    mse, mae, rsquared = calculate_metrics(fold_predictions, test_smiles_target)
    fold_mse = fold_mse + mse
    fold_mae = fold_mae + mae
    fold_rquared = fold_rquared + rsquared

training_validation_time = (time.time() - start_time_train_val) / 60
print("Training Time :", training_validation_time, " Minutes")

fold_metrics = {"average_fold_mse": fold_mse / number_of_folds,
                "average_fold_mae": fold_mae / number_of_folds,
                "average_fold_rquared": fold_rquared / number_of_folds}

final_dict = {}
for i in range(number_of_folds):
    final_dict['fold ' + str(i) + ' mse'] = all_train_metrics[i]['training_mse']

f = pd.DataFrame.from_dict(final_dict)
f['fold mean'] = f.mean(axis=1)
average_mse = f['fold mean'].tolist()

fold_metrics['average_epoch_mse'] = average_mse

identifier_results_plot = identifier + "_train_plots"
identifier_train_val_metrics = identifier + "_train_metrics.csv"
save_dict(fold_metrics, identifier_train_val_metrics)

start_time_test = time.time()
all_models_predictions = []
all_models_metrics = []

for fold in range(number_of_folds):
    print("making fold ", fold, " predictions")
    test_predictions = test_model(test_dataloader, all_networks[fold])
    all_models_predictions.append(test_predictions)

##calculate the average to create test_predictions_and_target, metrics_dict_test
smiles_target = test['docking_score'].tolist()
metrics_dict_test, test_predictions_and_target = create_test_metrics(all_models_predictions, smiles_target, number_of_folds, TESTING_SIZE)
test_time = (time.time() - start_time_test) / 60
print("Testing Time :", test_time, " Minutes")

identifier_test_scatter = identifier + "_test_scatter_plot"
identifier_test_heat_map = identifier + "_test_heat_map"
identifier_test_metrics = identifier + "_test_metrics.csv"
predictions_scatter_plot(test_predictions_and_target,
                         identifier_test_scatter, identifier)
predictions_heat_map(test_predictions_and_target,
                     identifier_test_heat_map, identifier)
save_dict(metrics_dict_test, identifier_test_metrics)

project_info_dict = {"training_size": [TRAINING_SIZE], "testing_size": [TESTING_SIZE],
                     str(number_of_folds) + " fold_validation_time": [training_validation_time], "testing_time": [test_time]}

identifier_project_info = identifier + "_project_info.csv"
save_dict(project_info_dict, identifier_project_info)
