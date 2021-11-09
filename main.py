import pandas as pd
import argparse
import warnings
import os
from other_models import OtherModels
from swfit_dock import SwiftDock

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="train code for training a network to estimate depth", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--targets", type=str, help="specify the target protein to ", nargs='+')
parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs='+')
parser.add_argument("--training_sizes", type=int, help="number of epochs", nargs='+')
args = parser.parse_args()

descriptors_dictionary = {'mac': [167, 'mac_keys_fingerprints(smile)'],
                          'onehot': [3500, 'one_hot_encode(smile)'],
                          'morgan_onehot_mac': [4691, 'morgan_fingerprints_mac_and_one_hot(smile)'],
                          'morgan_onehot_mac_circular': [4755, 'morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile)']}

descriptors_dictionary_command_line = {}
for desc in args.descriptors:
    if desc in descriptors_dictionary:
        descriptors_dictionary_command_line[desc] = [descriptors_dictionary[desc][0], descriptors_dictionary[desc][1]]

training_sizes = args.training_sizes
targets = args.targets
print('training_sizes', training_sizes)
print('targets', targets)
print('descriptors_dictionary_command_line', descriptors_dictionary_command_line)
training_metrics_dir = 'Results/training_metrics/'
testing_metrics_dir = 'Results/testing_metrics/'
test_predictions_dir = 'Results/test_predictions/'
project_info_dir = 'Results/project_info/'
dataset_dir = "Datasets/"
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
    for target in targets:
        for key, value in descriptors_dictionary_command_line.items():
            for size in training_sizes:
                identifier = f"swift_dock_{target}_{key}_{str(size)}"
                print('identifier ', identifier)
                numb_of_features = value[0]
                descriptor = value[1]
                path_to_csv_file = f"{dataset_dir}/{target}.csv"
                training_size = size
                number_of_folds = 2
                testing_size = 30
                training_size = training_size * number_of_folds
                feature_dim = numb_of_features
                train_models(training_metrics_dir=training_metrics_dir, testing_metrics_dir=testing_metrics_dir,
                             test_predictions_dir=test_predictions_dir, project_info_dir=project_info_dir,
                             target_path=path_to_csv_file, train_size=training_size, test_size=testing_size,
                             identifier=identifier, number_of_folds=number_of_folds, descriptor=descriptor,
                             feature_dim=numb_of_features)

    # for target, target_length in targets_list.items():
    #     for regressor_id, regressor in regressors_dict.items():
    #         for data_file, data_dim in dimensions.items():
    #             for size in train_sizes:
    #                 data_set_path = f"{dataset_dir}/{target}{data_file}.dat"
    #                 identifier = f"{regressor_id}{target}_{data_file}_{str(size)}"
    #                 print('data_set_path ', data_set_path)
    #                 all_data = np.memmap(data_set_path, dtype=np.float32, shape=(target_length, data_dim))
    #                 number_of_folds = 2
    #                 TRAINING_SIZE = size * number_of_folds
    #                 if TRAINING_SIZE > 99999:
    #                     TESTING_SIZE = len(X) - TRAINING_SIZE
    #                 else:
    #                     if target == 'Target 2':
    #                         TESTING_SIZE = 1400000
    #                     else:
    #                         TESTING_SIZE = 3000000
    #
    #                 train_ml(training_metrics_dir=training_metrics_dir, testing_metrics_dir=testing_metrics_dir,
    #                          test_predictions_dir=test_predictions_dir, project_info_dir=project_info_dir,
    #                          all_data=all_data, train_size=TRAINING_SIZE, test_size=TESTING_SIZE, identifier=identifier,
    #                          number_of_folds=number_of_folds, regressor=regressor)
