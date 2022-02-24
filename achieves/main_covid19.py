from os.path import exists

import pandas as pd
import argparse
import os
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

number_of_folds = 4

training_metrics_dir = '../results/training_metrics/'
testing_metrics_dir = '../results/testing_metrics/'
test_predictions_dir = '../results/test_predictions/'
project_info_dir = '../results/project_info/'
dataset_dir = "../datasets"
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


if __name__ == '__main__':
    path_to_csv_file = f"{dataset_dir}/{targets_swift_dock[0]}.csv"
    data = pd.read_csv(path_to_csv_file)
    all_targets = list(data.columns)
    all_targets.remove('TITLE')
    all_targets.remove('SMILES')

    for target in all_targets:
        target_data = data[['SMILES', target]]
        target_data.columns = ['smile', 'docking_score']
        target_data = target_data.dropna()
        target_data = target_data[target_data['smile'].map(len) <= 60]
        target_data = target_data[target_data['docking_score'] < 0.2]
        target = target.replace('_', '').replace('-', '')
        for key, value in descriptors_dictionary_command_line.items():
            for size in training_sizes_swift_dock:
                identifier = f"swift_dock_{target}_{key}_{str(size)}"
                already_exist_checker = f"{project_info_dir}{identifier}_project_info.csv"
                if exists(already_exist_checker):
                    continue
                logger.info(f"Identifier {identifier}")
                num_of_features = value[0]
                descriptor = value[1]
                item_training_size = size * number_of_folds
                if size > 500000:
                    item_testing_size = len(target_data) - item_training_size
                else:
                    item_testing_size = 500000

                train_models(training_metrics_dir=training_metrics_dir, testing_metrics_dir=testing_metrics_dir,
                             test_predictions_dir=test_predictions_dir, project_info_dir=project_info_dir,
                             target_path=target_data, train_size=item_training_size, test_size=item_testing_size,
                             identifier=identifier, number_of_folds=number_of_folds, descriptor=descriptor,
                             feature_dim=num_of_features)


    # for target in targets_swift_dock:
    #     path_to_csv_file = f"{dataset_dir}/{targets_swift_dock[0]}.csv"
    #     target_data = pd.read_csv(path_to_csv_file)
    #     target_data = target_data.dropna()
    #     for key, value in descriptors_dictionary_command_line.items():
    #         for size in training_sizes_swift_dock:
    #             identifier = f"swift_dock_{target}_{key}_{str(size)}"
    #             logger.info(f"Identifier {identifier}")
    #             num_of_features = value[0]
    #             descriptor = value[1]
    #             item_training_size = size * number_of_folds
    #
    #             if item_training_size > 99999:
    #                 item_testing_size = len(target_data) - item_training_size
    #             else:
    #                 if target == 'target2':
    #                     item_testing_size = 1400000
    #                 else:
    #                     item_testing_size = 3000000
    #
    #             train_models(training_metrics_dir=training_metrics_dir, testing_metrics_dir=testing_metrics_dir,
    #                          test_predictions_dir=test_predictions_dir, project_info_dir=project_info_dir,
    #                          target_path=target_data, train_size=item_training_size, test_size=item_testing_size,
    #                          identifier=identifier, number_of_folds=number_of_folds, descriptor=descriptor,
    #                          feature_dim=num_of_features)

