import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import pickle

from create_fingerprint_data import create_features
from swift_dock_logger import swift_dock_logger
from utils import calculate_metrics, create_test_metrics, create_fold_predictions_and_target_df, save_dict
from smiles_featurizers import morgan_fingerprints_mac_and_one_hot, mac_keys_fingerprints, one_hot_encode

logger = swift_dock_logger()


class OtherModels:
    def __init__(self, training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,
                 all_data, train_size, test_size, val_size, identifier, number_of_folds, regressor,
                 serialized_models_path, descriptor):
        self.all_data = all_data
        self.training_metrics_dir = training_metrics_dir
        self.testing_metrics_dir = testing_metrics_dir
        self.test_predictions_dir = test_predictions_dir
        self.project_info_dir = project_info_dir
        self.train_size = train_size
        self.test_size = test_size
        self.val_size = val_size
        self.identifier = identifier
        self.number_of_folds = number_of_folds
        self.regressor = regressor
        self.serialized_models_path = serialized_models_path
        self.descriptor = descriptor
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.x_val = None
        self.y_val = None
        self.cross_validation_metrics = None
        self.all_regressors = None
        self.test_metrics = None
        self.test_predictions_and_target_df = None
        self.cross_validation_time = None
        self.test_time = None
        self.train_time = None
        self.single_regressor = None

    def split_data(self, cross_validate):
        if cross_validate:
            self.x, self.y = self.all_data[:, :-1], self.all_data[:, -1]

            self.x_train = self.x[:self.train_size]
            self.y_train = self.y[:self.train_size]

            self.x_val = self.x[self.train_size:self.train_size + self.val_size]
            self.y_val = self.y[self.train_size:self.train_size + self.val_size]

            self.x_test = self.x[self.train_size + self.val_size:self.train_size + self.val_size + self.test_size]
            self.y_test = self.y[self.train_size + self.val_size:self.train_size + self.val_size + self.test_size]
        else:
            self.x, self.y = self.all_data[:, :-1], self.all_data[:, -1]
            self.x_train = self.x[0:self.train_size]
            self.y_train = self.y[0:self.train_size]
            self.x_test = self.x[self.train_size:self.train_size + self.test_size]
            self.y_test = self.y[self.train_size:self.train_size + self.test_size]

    def train(self):
        logger.info(f"Training has started for {self.identifier}")
        start_time_train = time.time()
        rg = self.regressor()  # Create an instance of the regressor
        rg.fit(self.x_train, self.y_train)
        self.train_time = (time.time() - start_time_train) / 60
        self.single_regressor = rg
        identifier_model_path = f"{self.serialized_models_path}{self.identifier}_model.pkl"
        descriptor_dict = {'descriptor': self.descriptor}
        with open(identifier_model_path, 'wb') as file:
            pickle.dump((rg, descriptor_dict), file)
        logger.info(f"Training is Done! {self.identifier}")

    def diagnose(self):
        logger.info(f"Validation has started for {self.identifier}")
        kf = KFold(n_splits=self.number_of_folds)
        kf.get_n_splits(self.x_val)
        regressors_list = []
        train_metrics = {'average_fold_mse': [], 'average_fold_mae': [], 'average_fold_rsquared': []}
        start_time_train = time.time()
        for big_index, small_index in kf.split(self.x_val):
            x_train_fold, x_test_fold = self.x_val[small_index], self.x_val[big_index]
            y_train_fold, y_test_fold = self.y_val[small_index], self.y_val[big_index]
            rg = self.regressor()  # Create an instance of the regressor
            rg.fit(x_train_fold, y_train_fold)
            regressors_list.append(rg)
            predictions = rg.predict(x_test_fold)
            mse, mae, rsquared = calculate_metrics(predictions, y_test_fold)
            train_metrics['average_fold_mse'].append(mse)
            train_metrics['average_fold_mae'].append(mae)
            train_metrics['average_fold_rsquared'].append(rsquared)
        self.cross_validation_time = (time.time() - start_time_train) / 60
        average_fold_mse = sum(train_metrics['average_fold_mse']) / len(train_metrics['average_fold_mse'])
        average_fold_mae = sum(train_metrics['average_fold_mae']) / len(train_metrics['average_fold_mae'])
        average_fold_r2 = sum(train_metrics['average_fold_rsquared']) / len(train_metrics['average_fold_rsquared'])
        train_metrics = {'average_fold_mse': [average_fold_mse], 'average_fold_mae': [average_fold_mae],
                         'average_fold_rsquared': [average_fold_r2]}
        self.cross_validation_metrics = train_metrics
        self.all_regressors = regressors_list
        identifier_train_val_metrics = f"{self.training_metrics_dir}{self.identifier}_cross_validation_metrics.csv"
        save_dict(self.cross_validation_metrics, identifier_train_val_metrics)

    def test(self):
        logger.info(f"Testing has started for {self.identifier}")
        all_models_predictions = []
        start_time_test = time.time()
        fold_predictions = self.single_regressor.predict(self.x_test)
        all_models_predictions.append(fold_predictions)
        self.test_time = (time.time() - start_time_test) / 60
        metrics_dict_test = create_test_metrics(all_models_predictions, self.y_test, 1)
        predictions_and_target_df = create_fold_predictions_and_target_df(all_models_predictions, self.y_test,
                                                                          1, self.test_size)
        self.test_metrics = metrics_dict_test
        self.test_predictions_and_target_df = predictions_and_target_df
        logger.info(f"Testing is Done! {self.identifier}")
        return all_models_predictions

    def save_results(self):
        identifier_test_metrics = f"{self.testing_metrics_dir}{self.identifier}_test_metrics.csv"
        save_dict(self.test_metrics, identifier_test_metrics)
        identifier_test_pred_target_df = f"{self.test_predictions_dir}{self.identifier}_test_predictions.csv"
        self.test_predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)
        project_info_dict = {"training_size": [self.train_size], "testing_size": [self.test_size],
                             str(self.number_of_folds) + " fold_validation_time": [self.cross_validation_time],
                             'training_time': self.train_time,
                             "testing_time": [self.test_time]}
        identifier_project_info = f"{self.project_info_dir}{self.identifier}_project_info.csv"
        save_dict(project_info_dict, identifier_project_info)
        logger.info(f"Saving done started for {self.identifier}")

    @staticmethod
    def inference(input_path, output_path, model_path):
        smiles = pd.read_csv(input_path)['smile'].tolist()
        tmp_path = '../../datasets/tmp.csv'
        logger.info('Inference has started...')
        df = pd.read_csv(input_path)
        df['docking_score'] = 0
        df.to_csv(tmp_path, index=False)
        # Load the model
        with open(model_path, 'rb') as file:
            pickle_model, descriptor_dict = pickle.load(file)
        descriptor = descriptor_dict['descriptor']
        info = {
            'onehot': [3500, one_hot_encode],
            'morgan_onehot_mac': [4691, morgan_fingerprints_mac_and_one_hot],
            'mac': [167, mac_keys_fingerprints]
        }
        dimensions_ml_models = {'onehot': 3500 + 1, 'morgan_onehot_mac': 4691 + 1,
                                'mac': 167 + 1}
        new_dict = {descriptor: info[descriptor]}
        create_features(['tmp'], new_dict)
        os.remove(tmp_path)
        data_set_path = f'../../datasets/tmp_{descriptor}.dat'
        data = np.memmap(data_set_path, dtype=np.float32)
        target_length = data.shape[0] // dimensions_ml_models[descriptor]
        data = data.reshape((target_length, dimensions_ml_models[descriptor]))
        x, y = data[:, :-1], data[:, -1]
        predictions = pickle_model.predict(x)
        results_dict = {"smile": smiles, "docking_score": predictions}
        identifier_project_info = f"{output_path}/results.csv"
        save_dict(results_dict, identifier_project_info)
        logger.info('Inference is finished')
