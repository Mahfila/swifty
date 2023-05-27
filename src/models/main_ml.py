import numpy as np
import argparse
import os
from ml_models import OtherModels
from src.utils.swift_dock_logger import swift_dock_logger

logger = swift_dock_logger()

parser = argparse.ArgumentParser(description="train code for training a network to estimate depth", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--targets", type=str, help="specify the target protein to ", nargs='+')
parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs='+')
parser.add_argument("--training_sizes", type=int, help="number of epochs", nargs='+')
args = parser.parse_args()


##### Models  Arugments
regressors_dict_ml_models = {'decision_tree': 'DecisionTreeRegressor()', 'xgboost': 'XGBRegressor()', 'sgdreg': 'SGDRegressor()'}
dimensions_ml_models = {'onehot': 3500 + 1, 'morgan_onehot_mac_circular': 4755 + 1, 'morgan_onehot_mac': 4691 + 1,
                        'mac': 167 + 1}


training_sizes_ml = args.training_sizes
targets = args.targets

number_of_folds = 5

training_metrics_dir = '../../results/training_metrics/'
testing_metrics_dir = '../../results/testing_metrics/'
test_predictions_dir = '../../results/test_predictions/'
project_info_dir = '../../results/project_info/'
dataset_dir = "../../datasets/"
os.makedirs(training_metrics_dir, exist_ok=True)
os.makedirs(testing_metrics_dir, exist_ok=True)
os.makedirs(test_predictions_dir, exist_ok=True)
os.makedirs(project_info_dir, exist_ok=True)


def train_ml(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,
             all_data, train_size, test_size, identifier, number_of_folds, regressor):
    model = OtherModels(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir, all_data=all_data, train_size=train_size, test_size=test_size,
                        identifier=identifier, number_of_folds=number_of_folds, regressor=regressor)
    model.cross_validate()
    model.test()
    model.save_results()


if __name__ == '__main__':
    for target in targets:
        for regressor_id, regressor in regressors_dict_ml_models.items():
            for data_file, data_dim in dimensions_ml_models.items():
                for size in training_sizes_ml:
                    data_set_path = f"{dataset_dir}{target}_{data_file}.dat"
                    identifier = f"{regressor_id}_{target}_{data_file}_{str(size)}"
                    data = np.memmap(data_set_path, dtype=np.float32)
                    target_length = data.shape[0] // data_dim
                    data = data.reshape((target_length, data_dim))

                    training_size_ml = size * number_of_folds
                    testing_size_ml = (target_length - training_size_ml)

                    train_ml(training_metrics_dir=training_metrics_dir, testing_metrics_dir=testing_metrics_dir,
                             test_predictions_dir=test_predictions_dir, project_info_dir=project_info_dir,
                             all_data=data, train_size=training_size_ml, test_size=testing_size_ml, identifier=identifier,
                             number_of_folds=number_of_folds, regressor=regressor)
