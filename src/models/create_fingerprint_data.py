import argparse
import os
import time
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from src.utils.smiles_featurizers import morgan_fingerprints_mac_and_one_hot_descriptors_circular_fingerprint, mac_keys_fingerprints, one_hot_encode, morgan_fingerprints_mac_and_one_hot

from src.utils.swift_dock_logger import swift_dock_logger

parser = argparse.ArgumentParser(description="train code for training a network to estimate depth", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--targets", type=str, help="specify the target protein to ", nargs='+')
args = parser.parse_args()
targets = args.targets


logger = swift_dock_logger()
info = {
    'onehot': [3500, 'one_hot_encode(smile)'],
    'morgan_onehot_mac_circular': [4755, 'morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile)'],
    'morgan_onehot_mac': [4691, 'morgan_fingerprints_mac_and_one_hot(smile)'],
    'mac': [167, 'mac_keys_fingerprints(smile)']}

info = {
    'morgan_onehot_mac_circular': [4755, 'morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile)']}
dataset_dir = "../../datasets"


class Training(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict.iloc[idx]
        smile = str(data['smile'])
        score = data['docking_score']
        score = np.array(score, dtype='float32').reshape(1, )
        features = eval(descriptor)
        X_Y = np.concatenate((features, score))
        return X_Y


for target in targets:
    for key, item in info.items():
        fingerprint_name = key
        number_of_features = item[0] + 1  # + target
        descriptor = item[1]
        path_to_csv_file = f"{dataset_dir}/{target}.csv"
        data = pd.read_csv(path_to_csv_file)
        data = data.dropna()
        data = data[data['smile'].map(len) <= 60]

        smiles_data_train = Training(data)
        directory = f"{dataset_dir}/{target}_{fingerprint_name}.dat"
        data_set = np.memmap(directory, dtype=np.float32,
                             mode='w+', shape=(len(data), item[0] + 1))
        start_time_test = time.time()
        init = 0
        batch_size = 128
        train_dataloader = DataLoader(smiles_data_train, batch_size=batch_size, shuffle=True, num_workers=28)
        for i, data in enumerate(train_dataloader):
            numpy_data = data.numpy()
            data_set[init:init + numpy_data.shape[0], :] = numpy_data
            init = init + numpy_data.shape[0]
        create_time = (time.time() - start_time_test) / 60
        logger.info(f"Creating Time : {create_time} Minutes")
        del data_set
        time_dict = {'creation_time': create_time}
        # saving_dir = f"{dataset_dir}/{target}_{fingerprint_name}_creation_time.txt"
        # with open(saving_dir, 'w+') as file:
        #     file.write(str(time_dict))
