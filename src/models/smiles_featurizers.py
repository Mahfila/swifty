import deepchem as dc
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Crippen
from rdkit.Chem import Descriptors
from rdkit.Chem import Lipinski
from rdkit.Chem import MACCSkeys


def circular_fingerprint(smile):
    descriptor = dc.feat.CircularFingerprint(size=64, radius=5)
    features = descriptor.featurize(smile).reshape(64, )
    features = features.astype('float32')
    return features


def mac_keys_fingerprints(smile):
    mol = Chem.MolFromSmiles(smile)
    fingerprint = MACCSkeys.GenMACCSKeys(mol)
    list1 = []
    list1[:0] = fingerprint
    chars_array = np.array([list1])
    chars_array = chars_array.astype('float32')
    return chars_array.reshape(167)


def one_hot_encode(smile):
    descriptor = dc.feat.OneHotFeaturizer()
    encodings = descriptor.featurize(smile).reshape(3500, )
    encodings = encodings.astype('float32')
    return encodings


def morgan_fingerprints(smile):
    m1 = Chem.MolFromSmiles(smile)
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(m1, 1, nBits=1024).ToBitString()
    list1 = []
    list1[:0] = fingerprint
    chars_array = np.array([list1])
    chars_array = chars_array.astype('float32')
    return chars_array.reshape(1024)


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


def compute_descriptors(mol):
    descriptors = {}

    # Basic descriptors
    descriptors["mol_weight"] = Descriptors.MolWt(mol)
    descriptors["num_atoms"] = mol.GetNumAtoms()
    descriptors["num_bonds"] = mol.GetNumBonds()
    descriptors["num_rotatable_bonds"] = Descriptors.NumRotatableBonds(mol)

    # Lipinski's descriptors
    descriptors["num_h_donors"] = Lipinski.NumHDonors(mol)
    descriptors["num_h_acceptors"] = Lipinski.NumHAcceptors(mol)

    # Crippen's descriptors related to logP and molar refractivity
    descriptors["logp"] = Crippen.MolLogP(mol)
    descriptors["mr"] = Crippen.MolMR(mol)

    # Additional descriptors
    descriptors["tpsa"] = Descriptors.TPSA(mol)
    descriptors["num_rings"] = Descriptors.RingCount(mol)
    descriptors["num_aromatic_rings"] = Descriptors.NumAromaticRings(mol)
    descriptors["hall_kier_alpha"] = Descriptors.HallKierAlpha(mol)
    descriptors["fraction_csp3"] = Descriptors.FractionCSP3(mol)
    descriptors["num_nitrogens"] = Descriptors.NumValenceElectrons(mol) - Descriptors.NumValenceElectrons(
        Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#7]")))
    descriptors["num_oxygens"] = Descriptors.NumValenceElectrons(mol) - Descriptors.NumValenceElectrons(
        Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#8]")))
    descriptors["num_sulphurs"] = Descriptors.NumValenceElectrons(mol) - Descriptors.NumValenceElectrons(
        Chem.DeleteSubstructs(mol, Chem.MolFromSmarts("[#16]")))

    return descriptors
