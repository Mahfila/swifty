import torch
from torch.utils.data import Dataset
from smiles_featurizers import mac_keys_fingerprints, one_hot_encode, morgan_fingerprints_mac_and_one_hot


class DataGenerator(Dataset):
    def __init__(self, data_dict, descriptor):
        self.data_dict = data_dict
        self.descriptor = descriptor

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict.iloc[idx]
        smile = str(data['smile'])
        score = data['docking_score']
        features = eval(self.descriptor)
        features = torch.from_numpy(features.reshape(features.shape[0], 1))
        score = torch.tensor([score])
        return features, score


class InferenceDataGenerator(Dataset):
    def __init__(self, data_dict, descriptor):
        self.data_dict = data_dict
        self.descriptor = descriptor

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, idx):
        data = self.data_dict.iloc[idx]
        smile = str(data['smile'])
        features = eval(self.descriptor)
        features = torch.from_numpy(features.reshape(features.shape[0], 1))
        return features
