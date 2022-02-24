import time
from sklearn.model_selection import KFold
import numpy as np
import time
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn import tree
import pandas as pd

from src.utils.swift_dock_logger import swift_dock_logger
from src.utils.utils import calculate_metrics, create_test_metrics, create_fold_predictions_and_target_df, save_dict

logger = swift_dock_logger()


class OtherModels:
    def __init__(self, training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,
                 all_data, train_size, test_size, identifier, number_of_folds, regressor):
        self.all_data = all_data
        self.training_metrics_dir = training_metrics_dir
        self.testing_metrics_dir = testing_metrics_dir
        self.test_predictions_dir = test_predictions_dir
        self.project_info_dir = project_info_dir
        self.train_size = train_size
        self.test_size = test_size
        self.identifier = identifier
        self.number_of_folds = number_of_folds
        self.regressor = regressor
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.cross_validation_metrics = None
        self.all_regressors = None
        self.test_metrics = None
        self.test_predictions_and_target_df = None
        self.cross_validation_time = None
        self.test_time = None

    def cross_validate(self):
        logger.info(f"Training has started for {self.identifier}")
        self.x, self.y = self.all_data[:, :-1], self.all_data[:, -1]
        self.x_train = self.x[0:self.train_size]
        self.y_train = self.y[0:self.train_size]
        self.x_test = self.x[self.train_size:self.train_size + self.test_size]
        self.y_test = self.y[self.train_size:self.train_size + self.test_size]
        start_time_train = time.time()
        kf = KFold(n_splits=self.number_of_folds)
        kf.get_n_splits(self.x_train)
        regressors_list = []
        train_metrics = {'average_fold_mse': [], 'average_fold_mae': [], 'average_fold_rsquared': []}
        start_time_train = time.time()
        for big_index, small_index in kf.split(self.x_train):
            x_train_fold, x_test_fold = self.x_train[small_index], self.x_train[big_index]
            y_train_fold, y_test_fold = self.y_train[small_index], self.y_train[big_index]
            rg = eval(self.regressor)
            rg = rg.fit(x_train_fold, y_train_fold)
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
        train_metrics = {'average_fold_mse': [average_fold_mse], 'average_fold_mae': [average_fold_mae], 'train_rsquared': [average_fold_r2]}
        self.cross_validation_metrics = train_metrics
        self.all_regressors = regressors_list

    def test(self):
        logger.info(f"Testing has started for {self.identifier}")
        all_models_predictions = []
        start_time_test = time.time()
        for regressor_obj in self.all_regressors:
            fold_predictions = regressor_obj.predict(self.x_test)
            all_models_predictions.append(fold_predictions)
        self.test_time = (time.time() - start_time_test) / 60
        metrics_dict_test = create_test_metrics(all_models_predictions, self.y_test, self.number_of_folds, self.test_size)
        predictions_and_target_df = create_fold_predictions_and_target_df(all_models_predictions, self.y_test, self.number_of_folds, self.test_size)
        self.test_metrics = metrics_dict_test
        self.test_predictions_and_target_df = predictions_and_target_df

        return all_models_predictions

    def save_results(self):
        identifier_train_val_metrics = f"{self.training_metrics_dir}{self.identifier}_train_metrics.csv"
        save_dict(self.cross_validation_metrics, identifier_train_val_metrics)
        identifier_test_metrics = f"{self.testing_metrics_dir}{self.identifier}_test_metrics.csv"
        save_dict(self.test_metrics, identifier_test_metrics)
        identifier_test_pred_target_df = f"{self.test_predictions_dir}{self.identifier}_test_predictions.csv"
        self.test_predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)
        project_info_dict = {"training_size": [self.train_size], "testing_size": [self.test_size],
                             str(self.number_of_folds) + " fold_validation_time": [self.cross_validation_time], "testing_time": [self.test_time]}
        identifier_project_info = f"{self.project_info_dir}{self.identifier}_project_info.csv"
        save_dict(project_info_dict, identifier_project_info)
        logger.info(f"Saving done started for {self.identifier}")
