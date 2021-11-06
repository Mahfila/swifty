import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


def test_model(test_dataloader, net):
    smiles_prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            features, _ = data
            features = features.squeeze()
            features = features.unsqueeze(0)
            outputs = net(features)
            smiles_prediction.extend(outputs.squeeze().tolist())
            del features
    return smiles_prediction


def create_fold_predictions_and_target_df(fold_predictions, smiles_target, number_of_folds, test_size):
    all_preds = np.zeros((test_size, number_of_folds + 1))
    for i in range(number_of_folds):
        all_preds[:, i] = fold_predictions[i]

    all_preds[:, -1] = smiles_target
    columns = ['f' + str(i) for i in range(number_of_folds)]
    columns.append('target')
    predictions_and_target_df = pd.DataFrame(all_preds, columns=columns)
    return predictions_and_target_df


def create_test_metrics(fold_predictions, smiles_target, number_of_folds, test_size):
    all_folds_mse = 0
    all_folds_mae = 0
    all_folds_rsquared = 0
    for i in range(number_of_folds):
        mse, mae, rsquared = calculate_metrics(
            fold_predictions[i], smiles_target)
        all_folds_mse = all_folds_mse + mse
        all_folds_mae = all_folds_mae + mae
        all_folds_rsquared = all_folds_rsquared + rsquared
    metrics_dict_test = {"test_mse": [], "test_mae": [], "test_rsquared": []}
    metrics_dict_test["test_mse"].append((all_folds_mse / number_of_folds))
    metrics_dict_test["test_mae"].append((all_folds_mae / number_of_folds))
    metrics_dict_test["test_rsquared"].append((all_folds_rsquared / number_of_folds) * 100)
    return metrics_dict_test


def calculate_metrics(predictions, target):
    mse = mean_squared_error(target, predictions)
    mae = mean_absolute_error(target, predictions)
    rsquared = r2_score(target, predictions)
    return mse, mae, rsquared


def get_training_and_test_data(DATA, TRAINING_SIZE, TESTING_SIZE):
    X = DATA['smile']
    Y = DATA['docking_score']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=TRAINING_SIZE, test_size=TESTING_SIZE, random_state=42)
    train = pd.concat([X_train, y_train], axis=1)
    test = pd.concat([X_test, y_test], axis=1)
    return train, test


def save_dict(history, identifier):
    result_df = pd.DataFrame.from_dict(history)
    result_df.to_csv(identifier)


def predictions_heat_map(test_predictions_and_target, identifier_test_heat_map, identifier):
    predictions = test_predictions_and_target["predictions"]
    target = test_predictions_and_target["target"]
    fig, ax1 = plt.subplots(1, 1)
    fig.set_dpi(500)
    fig.suptitle("Test Heat Map " + identifier, fontsize=10)
    ax1.set_xlabel('Target', fontsize=10)
    ax1.set_ylabel('Predictions', fontsize=10)
    plt.hexbin(target, predictions, gridsize=100, bins='log')
    fig.savefig(identifier_test_heat_map)


def get_smiles_dict(path_to_all_smiles):
    all_strings = ""
    f = open(path_to_all_smiles, "r")
    for x in f:
        all_strings = all_strings + x

    chars = tuple(set(all_strings))
    int2char = dict(enumerate(chars))
    char2int = {ch: ii for ii, ch in int2char.items()}
    dict_size = len(char2int)
    return int2char, char2int, dict_size
