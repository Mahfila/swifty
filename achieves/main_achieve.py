import pandas as pd
import argparse
import os
from other_models import OtherModels
from src.models.swift_dock import SwiftDock
from src.utils.swift_dock_logger import swift_dock_logger

logger = swift_dock_logger()

parser = argparse.ArgumentParser(description="train code for training a network to estimate depth", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--targets", type=str, help="specify the target protein to ", nargs='+')
parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs='+')
parser.add_argument("--training_sizes", type=int, help="number of epochs", nargs='+')
args = parser.parse_args()

#####Swift Dock Arugments

descriptors_dictionary = {'mac': [167, 'mac_keys_fingerprints(smile)'],
                          'onehot': [3500, 'one_hot_encode(smile)'],
                          'morgan_onehot_mac': [4691, 'morgan_fingerprints_mac_and_one_hot(smile)'],
                          'morgan_onehot_mac_circular': [4755, 'morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile)']}

training_sizes_swift_dock = args.training_sizes
targets_swift_dock = args.targets
logger.info(f"training_sizes {training_sizes_swift_dock}")
logger.info(f"targets {targets_swift_dock}", )

descriptors_dictionary_command_line = {}
for desc in args.descriptors:
    if desc in descriptors_dictionary:
        descriptors_dictionary_command_line[desc] = [descriptors_dictionary[desc][0], descriptors_dictionary[desc][1]]
print('descriptors_dictionary_command_line', descriptors_dictionary_command_line)

##### Other Models  Arugments
regressors_dict_ml_models = {'decision_tree': 'DecisionTreeRegressor()', 'xgboost': 'XGBRegressor()', 'sgdreg': 'SGDRegressor()'}
targets_list_ml_models = {'target1': 3437838}
dimensions_ml_models = {'onehot': 3500 + 1, 'morgan_onehot_mac_circular': 4755 + 1, 'morgan_onehot_mac': 4691 + 1,
                        'mac': 167 + 1}
training_sizes_ml = [7000, 10000, 20000, 50000, 100000, 350000]

number_of_folds = 5

training_metrics_dir = '../results/training_metrics'
testing_metrics_dir = '../results/testing_metrics'
test_predictions_dir = '../results/test_predictions'
project_info_dir = '../results/project_info'
dataset_dir = "../datasets/"
os.makedirs(training_metrics_dir, exist_ok=True)
os.makedirs(testing_metrics_dir, exist_ok=True)
os.makedirs(test_predictions_dir, exist_ok=True)
os.makedirs(project_info_dir, exist_ok=True)


def train_models(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,
                 target_path, train_size, test_size, identifier, number_of_folds, descriptor, feature_dim):
    model = SwiftDock(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir, target_path=target_path, train_size=train_size, test_size=test_size,
                      identifier=identifier, number_of_folds=number_of_folds, descriptor=descriptor,
                      feature_dim=feature_dim)
    model.cross_validate()
    model.test()
    model.save_results()


def train_ml(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,
             all_data, train_size, test_size, identifier, number_of_folds, regressor):
    model = OtherModels(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir, all_data=all_data, train_size=train_size, test_size=test_size,
                        identifier=identifier, number_of_folds=number_of_folds, regressor=regressor)
    model.cross_validate()
    model.test()
    model.save_results()


if __name__ == '__main__':
    for target, target_length in targets_list_ml_models.items():
        for regressor_id, regressor in regressors_dict_ml_models.items():
            for data_file, data_dim in dimensions_ml_models.items():
                for size in training_sizes_ml:
                    data_set_path = f"{dataset_dir}/{target}{data_file}.dat"
                    identifier = f"{regressor_id}{target}_{data_file}_{str(size)}"
                    all_data = np.memmap(data_set_path, dtype=np.float32, shape=(target_length, data_dim))

                    training_size_ml = size * number_of_folds
                    if training_size_ml > 99999:
                        testing_size_ml = len(all_data) - training_size_ml
                    else:
                        if target == 'target2':
                            testing_size_ml = 1400000
                        else:
                            testing_size_ml = 3000000

                    train_ml(training_metrics_dir=training_metrics_dir, testing_metrics_dir=testing_metrics_dir,
                             test_predictions_dir=test_predictions_dir, project_info_dir=project_info_dir,
                             all_data=all_data, train_size=training_size_ml, test_size=testing_size_ml, identifier=identifier,
                             number_of_folds=number_of_folds, regressor=regressor)
