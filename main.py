import os
import time
import pandas as pd
from testing import test_models
from utils import get_training_and_test_data, save_dict
from trainer import trainer
import argparse
import warnings
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser(description="train code for training a network to estimate depth ", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--targets", type=str, help="specify the target protein to ", nargs='+')
parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs='+')
parser.add_argument("--training_size", type=int, help="number of epochs", nargs='+')
args = parser.parse_args()


targets = args.targets
info = {'mac': [167, 'mac_keys_fingerprints(smile)'],
        'onehot': [3500, 'one_hot_encode(smile)'],
        'morgan_onehot_mac': [4691, 'morgan_fingerprints_mac_and_one_hot(smile)'],
        'morgan_onehot_mac_circular': [4755, 'morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile)']}

info2 = {}
for desc in args.descriptors:
    if desc in info:
        info2[desc] = [info[desc][0], info[desc][1]]

print('info2', info2)

train_size = args.training_size
training_metrics_dir = 'Results/training_metrics/'
testing_metrics_dir = 'Results/testing_metrics/'
test_predictions_dir = 'Results/test_predictions/'
project_info_dir = 'Results/project_info/'
dataset_dir = "Datasets/"
os.makedirs(training_metrics_dir, exist_ok=True)
os.makedirs(testing_metrics_dir, exist_ok=True)
os.makedirs(test_predictions_dir, exist_ok=True)
os.makedirs(project_info_dir, exist_ok=True)

for target in targets:
    for key, value in info2.items():
        for size in train_size:
            identifier = f"{target}_{key}_{str(size)}"
            print('identifier ', identifier)
            numb_of_features = value[0]
            descriptor = value[1]
            PATH_TO_CSV = f"{dataset_dir}/{target}.csv"
            DATA = pd.read_csv(PATH_TO_CSV)
            data_set_size = len(DATA)
            TRAINING_SIZE = size
            number_of_folds = 5
            TESTING_SIZE = 30
            TRAINING_SIZE = TRAINING_SIZE * number_of_folds
            feature_dim = numb_of_features
            train, test = get_training_and_test_data(DATA, TRAINING_SIZE, TESTING_SIZE)
            print("training size ", TRAINING_SIZE)
            print("testing size ", TESTING_SIZE)
            start_time_train_val = time.time()
            fold_metrics, all_networks = trainer(train=train, number_of_folds=number_of_folds, feature_dim=feature_dim, descriptor=descriptor, identifier=identifier)
            training_validation_time = (time.time() - start_time_train_val) / 60
            identifier_train_val_metrics = f"{training_metrics_dir}{identifier}_train_metrics.csv"
            save_dict(fold_metrics, identifier_train_val_metrics)
            start_time_test = time.time()
            metrics_dict_test, predictions_and_target_df = test_models(test=test, number_of_folds=number_of_folds, all_networks=all_networks, descriptor=descriptor, TESTING_SIZE=TESTING_SIZE)
            test_time = (time.time() - start_time_test) / 60
            identifier_test_metrics = f"{testing_metrics_dir}{identifier}_test_metrics.csv"
            save_dict(metrics_dict_test, identifier_test_metrics)
            identifier_test_pred_target_df = f"{test_predictions_dir}{identifier}_test_predictions.csv"
            predictions_and_target_df.to_csv(identifier_test_pred_target_df, index=False)
            project_info_dict = {"training_size": [TRAINING_SIZE], "testing_size": [TESTING_SIZE],
                                 str(number_of_folds) + " fold_validation_time": [training_validation_time], "testing_time": [test_time]}
            identifier_project_info = f"{project_info_dir}{identifier}_project_info.csv"
            save_dict(project_info_dict, identifier_project_info)
