from rdkit.Chem import MACCSkeys
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


def mac_keys_fingerprints(smile):
    mol = Chem.MolFromSmiles(smile)
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    list1 = []
    list1[:0] = fingerprint
    chars_array = np.array([list1])
    chars_array = chars_array.astype('float32')
    return chars_array.reshape((167))


def smile_fingerprint(smile):
    ms = [Chem.MolFromSmiles(smile)]
    fingerprint = Chem.RDKFingerprint(ms[0], 4, fpSize=2048).ToBitString()
    list1 = []
    list1[:0] = fingerprint
    chars_array = np.array([list1])
    chars_array = chars_array.astype('float32')
    return chars_array


def one_hot_encode(smile, char2int, dict_size):
    list_of_chars = [char2int[char] for char in smile]
    chars_array = np.array([list_of_chars])

    one_hot = np.zeros((chars_array.size, dict_size), dtype=np.float32)

    one_hot[np.arange(one_hot.shape[0]), chars_array.flatten()] = 1.

    one_hot = one_hot.reshape((*chars_array.shape, dict_size))

    one_hot = one_hot.squeeze()
    shape0 = one_hot.shape[0]
    shape1 = one_hot.shape[1]
    one_hot = one_hot.reshape((shape0 * shape1))
    max_lenth = 60 * 39  # max smile by dict size
    dim_difference = max_lenth - (shape0 * shape1)
    features = np.pad(one_hot, (0, dim_difference), 'constant')

    return features


def morgan_fingerprints(smile):
    m1 = Chem.MolFromSmiles(smile)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        m1, 1, nBits=1024).ToBitString()
    list1 = []
    list1[:0] = fingerprint
    chars_array = np.array([list1])
    chars_array = chars_array.astype('float32')
    return chars_array.reshape((1024))


def morgan_fingerprints_and_one_hot(smile, char2int, dict_size):
    one_hot_encoding = one_hot_encode(smile, char2int, dict_size)
    fingerprint_features = morgan_fingerprints(smile)
    features = np.concatenate((fingerprint_features, one_hot_encoding))
    return features


def morgan_fingerprints_mac_and_one_hot(smile, char2int, dict_size):
    fingerprint_features = morgan_fingerprints(smile)
    mac_features = mac_keys_fingerprints(smile)
    one_hot_encoding = one_hot_encode(smile, char2int, dict_size)
    features = np.concatenate((fingerprint_features, one_hot_encoding))
    features = np.concatenate((features, mac_features))
    return features
