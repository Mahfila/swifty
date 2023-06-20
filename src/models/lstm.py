import time

import torch
import torch.nn as nn
from data_generator import DataGenerator, InferenceDataGenerator
from model import AttentionNetwork
from trainer import train_model
from utils import get_training_and_test_data, test_model, calculate_metrics, create_test_metrics, \
    create_fold_predictions_and_target_df, save_dict, get_data_splits, inference
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import copy
from swift_dock_logger import swift_dock_logger

logger = swift_dock_logger()


class SwiftDock:
    def __init__(self, training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir, target_path,
                 train_size, test_size, val_size, identifier, number_of_folds, descriptor, feature_dim,
                 serialized_models_path, cross_validate):
        self.target_path = target_path
        self.training_metrics_dir = training_metrics_dir
        self.testing_metrics_dir = testing_metrics_dir
        self.test_predictions_dir = test_predictions_dir
        self.project_info_dir = project_info_dir
        self.train_size = train_size
        self.test_size = test_size
        self.identifier = identifier
        self.number_of_folds = number_of_folds
        self.feature_dim = feature_dim
        self.descriptor = descriptor
        self.serialized_models_path = serialized_models_path
        self.train_data = None
        self.test_data = None
        self.val_data = val_size
        self.cross_validation_metrics = None
        self.all_networks = None
        self.test_metrics = None
        self.test_predictions_and_target_df = None
        self.cross_validation_time = None
        self.test_time = None
        self.single_mode_time = None
        self.single_model = None
        self.cross_validate = cross_validate

    def split_data(self, cross_validate):
        if cross_validate:
            self.train_data, self.test_data, self.val_data = get_data_splits(self.target_path, self.train_size,
                                                                             self.test_size, self.val_data)
        else:
            self.train_data, self.test_data = get_training_and_test_data(self.target_path, self.train_size,
                                                                         self.test_size)

    def train(self):
        logger.info('Starting training...')
        identifier_model_path = f"{self.serialized_models_path}{self.identifier}_model.pt"
        smiles_data_train = DataGenerator(self.train_data, descriptor=self.descriptor)  # train
        train_dataloader = DataLoader(smiles_data_train, batch_size=32, shuffle=True, num_workers=8)
        criterion = nn.MSELoss()
        net = AttentionNetwork(self.feature_dim)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        number_of_epochs = 7
        start = time.time()
        model, _ = train_model(train_dataloader, net, criterion,
                               optimizer, number_of_epochs)
        self.single_mode_time = (time.time() - start) / 60
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'descriptor': self.descriptor,
                    'num_of_features': self.feature_dim}, identifier_model_path)
        self.single_model = model

    def diagnose(self):
        logger.info('Starting diagnosis...')
        all_train_metrics = []
        df_split = np.array_split(self.val_data, self.number_of_folds)
        all_networks = []
        fold_mse, fold_mae, fold_rsquared = 0, 0, 0
        number_of_epochs = 7
        start_time_train_val = time.time()
        for fold in range(self.number_of_folds):
            net = AttentionNetwork(self.feature_dim)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            temp_data = copy.deepcopy(df_split)
            temp_data.pop(fold)
            temp_data = pd.concat(temp_data)
            smiles_data_train = DataGenerator(df_split[fold], descriptor=self.descriptor)  # train
            train_dataloader = DataLoader(smiles_data_train, batch_size=32, shuffle=True, num_workers=8)
            fold_test_dataloader_class = DataGenerator(temp_data, descriptor=self.descriptor)
            fold_test_dataloader = DataLoader(fold_test_dataloader_class, batch_size=32, shuffle=False, num_workers=8)
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
            fold_rsquared = fold_rsquared + rsquared
        self.cross_validation_time = (time.time() - start_time_train_val) / 60
        cross_validation_metrics = {"average_fold_mse": fold_mse / self.number_of_folds,
                                    "average_fold_mae": fold_mae / self.number_of_folds,
                                    "average_fold_rsquared": fold_rsquared / self.number_of_folds}

        final_dict = {}
        for i in range(self.number_of_folds):
            final_dict['fold ' + str(i) + ' mse'] = all_train_metrics[i]['training_mse']
        f = pd.DataFrame.from_dict(final_dict)
        f['fold mean'] = f.mean(axis=1)
        average_mse = f['fold mean'].tolist()
        cross_validation_metrics['average_epoch_mse'] = average_mse
        self.cross_validation_metrics = cross_validation_metrics
        self.all_networks = all_networks

    def test(self):
        logger.info('Starting testing...')
        all_models_predictions = []
        smiles_data_test = DataGenerator(self.test_data, descriptor=self.descriptor)
        test_dataloader = DataLoader(smiles_data_test, batch_size=16, shuffle=False, num_workers=6)
        start_time_test = time.time()
        test_predictions = test_model(test_dataloader, self.single_model)
        all_models_predictions.append(test_predictions)
        self.test_time = (time.time() - start_time_test) / 60
        smiles_target = self.test_data['docking_score'].tolist()
        metrics_dict_test = create_test_metrics(all_models_predictions, smiles_target, 1)
        predictions_and_target_df = create_fold_predictions_and_target_df(all_models_predictions, smiles_target,
                                                                          1, self.test_size)
        self.test_metrics = metrics_dict_test
        self.test_predictions_and_target_df = predictions_and_target_df

    def save_results(self):
        if self.cross_validate:
            identifier_train_val_metrics = f"{self.training_metrics_dir}{self.identifier}_cross_validation_metrics.csv"
            save_dict(self.cross_validation_metrics, identifier_train_val_metrics)
        identifier_test_metrics = f"{self.testing_metrics_dir}{self.identifier}_test_metrics.csv"
        save_dict(self.test_metrics, identifier_test_metrics)
        identifier_test_pred_target_df = f"{self.test_predictions_dir}{self.identifier}_test_predictions.csv"
        self.test_predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)
        project_info_dict = {"training_size": [self.train_size], "testing_size": [self.test_size], 'training_time':self.single_mode_time,
                             str(self.number_of_folds) + " fold_validation_time": [self.cross_validation_time],
                             "testing_time": [self.test_time]}
        identifier_project_info = f"{self.project_info_dir}{self.identifier}_project_info.csv"
        save_dict(project_info_dict, identifier_project_info)
        logger.info('Training and Testing information has been saved.')

    @staticmethod
    def inference(input_path, output_path, model_path):
        logger.info('Inference has started...')
        smiles = pd.read_csv(input_path)['smile'].tolist()
        # Load the model
        checkpoint = torch.load(model_path)
        descriptor = checkpoint['descriptor']
        num_of_features = checkpoint['num_of_features']
        model = AttentionNetwork(num_of_features)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.eval()
        smiles_data_train = InferenceDataGenerator(pd.read_csv(input_path), descriptor=descriptor)  # train
        torch.set_num_threads(6)
        inference_dataloader = DataLoader(smiles_data_train, batch_size=32, shuffle=False, num_workers=6)
        predictions = inference(inference_dataloader, model)
        results_dict = {"smile": smiles, "docking_score": predictions}
        identifier_project_info = f"{output_path}/results.csv"
        save_dict(results_dict, identifier_project_info)
        logger.info('Inference is finished')
