from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import deepchem as dc


def CircularFingerprint(smile):
    featurizer = dc.feat.CircularFingerprint(size=64, radius=5)
    features = featurizer.featurize(smile).reshape(64, )
    features = features.astype('float32')
    return features


def mac_keys_fingerprints(smile):
    mol = Chem.MolFromSmiles(smile)
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    list1 = []
    list1[:0] = fingerprint
    chars_array = np.array([list1])
    chars_array = chars_array.astype('float32')
    return chars_array.reshape((167))


def one_hot_encode(smile):
    featurizer = dc.feat.OneHotFeaturizer()
    encodings = featurizer.featurize(smile).reshape(3500, )
    encodings = encodings.astype('float32')
    return encodings


def morgan_fingerprints(smile):
    m1 = Chem.MolFromSmiles(smile)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        m1, 1, nBits=1024).ToBitString()
    list1 = []
    list1[:0] = fingerprint
    chars_array = np.array([list1])
    chars_array = chars_array.astype('float32')
    return chars_array.reshape((1024))


def morgan_fingerprints_and_one_hot(smile):
    one_hot_encoding = one_hot_encode(smile)
    fingerprint_features = morgan_fingerprints(smile)
    features = np.concatenate((fingerprint_features, one_hot_encoding))
    return features


def morgan_fingerprints_mac_and_one_hot(smile):
    fingerprint_features = morgan_fingerprints(smile)
    mac_features = mac_keys_fingerprints(smile)
    one_hot_encoding = one_hot_encode(smile)
    features = np.concatenate((fingerprint_features, one_hot_encoding))
    features = np.concatenate((features, mac_features))
    return features


def morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile):
    fingerprint_features = morgan_fingerprints(smile)
    mac_features = mac_keys_fingerprints(smile)
    one_hot_encoding = one_hot_encode(smile)
    pubchem = CircularFingerprint(smile)
    features = np.concatenate((fingerprint_features, one_hot_encoding))
    features = np.concatenate((features, mac_features))
    features = np.concatenate((features, pubchem))
    return features
