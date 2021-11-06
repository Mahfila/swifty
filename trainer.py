import copy
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from data_generator import DataGenerator
from model import AttentionNetwork
from utils import test_model, save_dict, calculate_metrics
from smiles_featurizers import morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint,mac_keys_fingerprints,one_hot_encode,morgan_fingerprints_mac_and_one_hot


def train_model(train_dataloader, model, criterion, optimizer, number_of_epochs):
    metrics_dict = {"training_mse": []}
    mse_array_train = []

    for epoch in range(number_of_epochs):
        running_loss = 0
        for i_batch, data in enumerate(train_dataloader, 0):
            optimizer.zero_grad()
            features, target = data
            features = features.squeeze()
            features = features.unsqueeze(0)
            outputs = model(features).double()
            outputs = outputs.squeeze(0)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            del features, outputs
            running_loss += loss.item()
        mse_array_train.append(running_loss / len(train_dataloader))
        print("Epoch: {}/{}.. ".format(epoch + 1, number_of_epochs),
              "Training MSE: {:.3f}.. ".format(running_loss / len(train_dataloader)))

    metrics_dict["training_mse"] = mse_array_train
    print("Finished training.")
    return model, metrics_dict


def trainer(train, number_of_folds, feature_dim, descriptor, identifier):
    start_time_train_val = time.time()
    all_train_metrics = []
    df_split = np.array_split(train, number_of_folds)
    all_networks = []
    all_fold_test_mse = []
    fold_mse, fold_mae, fold_rquared = 0, 0, 0
    number_of_epochs = 9
    for fold in range(number_of_folds):
        net = AttentionNetwork(feature_dim)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        temp_data = copy.deepcopy(df_split)
        temp_data.pop(fold)
        temp_data = pd.concat(temp_data)
        smiles_data_train = DataGenerator(df_split[fold], descriptor=descriptor)  # train
        print('len of training ', len(smiles_data_train))
        train_dataloader = DataLoader(smiles_data_train, batch_size=128, shuffle=True, num_workers=6)
        fold_test_dataloader_class = DataGenerator(temp_data, descriptor=descriptor)
        print('len of testing ', len(fold_test_dataloader_class))
        fold_test_dataloader = DataLoader(fold_test_dataloader_class, batch_size=128, shuffle=False, num_workers=6)
        criterion = nn.MSELoss()
        # training
        model, metrics_dict = train_model(train_dataloader, net, criterion,
                                          optimizer, number_of_epochs)
        all_networks.append(model)
        all_train_metrics.append(metrics_dict)

        # Validate
        fold_predictions = test_model(fold_test_dataloader, model)
        test_smiles_target = temp_data['docking_score'].tolist()
        mse, mae, rsquared = calculate_metrics(fold_predictions, test_smiles_target)
        fold_mse = fold_mse + mse
        fold_mae = fold_mae + mae
        fold_rquared = fold_rquared + rsquared

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
    return fold_metrics, all_networks
