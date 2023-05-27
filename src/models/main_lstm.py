import pandas as pd
import argparse
import os
from src.models.lstm import SwiftDock
from src.utils.swift_dock_logger import swift_dock_logger

logger = swift_dock_logger()

parser = argparse.ArgumentParser(description="train code for training a network to estimate depth",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--targets", type=str, help="specify the target protein to ", nargs='+')
parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs='+')
parser.add_argument("--training_sizes", type=int, help="number of epochs", nargs='+')
args = parser.parse_args()

#####Swift Dock Arguments

descriptors_dictionary = {'mac': [167, 'mac_keys_fingerprints(smile)'],
                          'onehot': [3500, 'one_hot_encode(smile)'],
                          'morgan_onehot_mac': [4691, 'morgan_fingerprints_mac_and_one_hot(smile)']}

training_sizes_swift_dock = args.training_sizes
targets_swift_dock = args.targets

descriptors_dictionary_command_line = {}
for desc in args.descriptors:
    if desc in descriptors_dictionary:
        descriptors_dictionary_command_line[desc] = [descriptors_dictionary[desc][0], descriptors_dictionary[desc][1]]

number_of_folds = 5

training_metrics_dir = '../../results/training_metrics/'
testing_metrics_dir = '../../results/testing_metrics/'
test_predictions_dir = '../../results/test_predictions/'
project_info_dir = '../../results/project_info/'
dataset_dir = "../../datasets"
os.makedirs(training_metrics_dir, exist_ok=True)
os.makedirs(testing_metrics_dir, exist_ok=True)
os.makedirs(test_predictions_dir, exist_ok=True)
os.makedirs(project_info_dir, exist_ok=True)


def train_models(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,
                 target_path, train_size, test_size, identifier, number_of_folds, descriptor, feature_dim):
    model = SwiftDock(training_metrics_dir, testing_metrics_dir, test_predictions_dir, project_info_dir,
                      target_path=target_path, train_size=train_size, test_size=test_size,
                      identifier=identifier, number_of_folds=number_of_folds, descriptor=descriptor,
                      feature_dim=feature_dim)
    model.cross_validate()
    model.test()
    model.save_results()


if __name__ == '__main__':
    for target in targets_swift_dock:
        for key, value in descriptors_dictionary_command_line.items():
            for size in training_sizes_swift_dock:
                identifier = f"lstm_{target}_{key}_{str(size)}"
                logger.info(f"Identifier {identifier}")
                num_of_features = value[0]
                descriptor = value[1]
                path_to_csv_file = f"{dataset_dir}/{target}.csv"
                data_all = pd.read_csv(path_to_csv_file)
                data_all = data_all.dropna()
                target_len = len(data_all)
                item_training_size = size * number_of_folds
                item_testing_size = (target_len - item_training_size)

                train_models(training_metrics_dir=training_metrics_dir, testing_metrics_dir=testing_metrics_dir,
                             test_predictions_dir=test_predictions_dir, project_info_dir=project_info_dir,
                             target_path=data_all, train_size=item_training_size, test_size=item_testing_size,
                             identifier=identifier, number_of_folds=number_of_folds, descriptor=descriptor,
                             feature_dim=num_of_features)
