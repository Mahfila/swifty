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
            outputs = net(features)
            smiles_prediction.extend(outputs.squeeze().tolist())
            del features
    return smiles_prediction



def create_test_metrics(fold_predictions,smiles_target,number_of_folds,test_size):
    all_preds = np.zeros((test_size,number_of_folds))
    for i in range(number_of_folds):
        all_preds[:,i] = fold_predictions[i]
    mean_of_predictions = all_preds.mean(axis=1)
    mse, mae, rsquared = calculate_metrics(mean_of_predictions, smiles_target)
    metrics_dict_test = {"test_mse": [], "test_mae": [], "test_rsquared": []}
    test_predictions_and_target = {"predictions": [], "target": []}
    test_predictions_and_target["predictions"] = mean_of_predictions
    test_predictions_and_target["target"] = smiles_target
    metrics_dict_test["test_mse"].append(mse)
    metrics_dict_test["test_mae"].append(mae)
    metrics_dict_test["test_rsquared"] = rsquared
    return metrics_dict_test,test_predictions_and_target


def calculate_metrics(predictions, target):
    mse = mean_squared_error(target, predictions)
    mae = mean_absolute_error(target, predictions)
    rsquared = r2_score(target, predictions)
    return mse, mae, rsquared


def get_training_and_test_data(DATA, TRAINING_SIZE, TESTING_SIZE):
    train = DATA.sample(TRAINING_SIZE)
    test = pd.concat([train, DATA]).drop_duplicates(keep=False)
    test = test.sample(TESTING_SIZE)
    return train, test


def get_data_dictionaries(DATA, TRAINING_SIZE, TESTING_SIZE):
    DATA = DATA.dropna()
    del DATA["Unnamed: 0"]
    train = DATA.sample(TRAINING_SIZE)
    test = pd.concat([train, DATA]).drop_duplicates(keep=False)
    test = test.sample(TESTING_SIZE)
    whole_train_indexes = [i for i in range(TRAINING_SIZE)]
    whole_test_indexes = [i for i in range(test.shape[0])]
    train.reset_index(inplace=True)
    test.reset_index(inplace=True)
    train["indexes"] = whole_train_indexes
    test["indexes"] = whole_test_indexes

    train_dict = train.set_index('indexes').T.to_dict('list')
    test_dict = test.set_index('indexes').T.to_dict('list')

    return train_dict, test_dict


def save_dict(history, identifier):
    result_df = pd.DataFrame.from_dict(history)
    result_df.to_csv(identifier)


def plot_history(metrics_dict, identifier, identifier_results_plot):
    fig, ax1 = plt.subplots(1)
    fig.set_dpi(500)
    # Plot 1
    fig.suptitle('Training Loss-' + identifier)
    ax1.plot(metrics_dict["training_mse"], label="training Loss")
    ax1.legend(loc='best')
    ax1.legend()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.tick_params(axis='x')
    ax1.tick_params(axis='y')

    plt.show()
    fig.savefig(identifier_results_plot)


def predictions_scatter_plot(test_predictions_and_target, identifier_test_scatter, identifier):
    predictions = test_predictions_and_target["predictions"]
    target = test_predictions_and_target["target"]
    fig, ax1 = plt.subplots(1, 1)
    fig.set_dpi(500)
    fig.suptitle("Test Scatter Plot " + identifier, fontsize=10)
    ax1.set_xlabel('Target', fontsize=10)
    ax1.set_ylabel('Predictions', fontsize=10)
    plt.scatter(target, predictions)
    fig.savefig(identifier_test_scatter)


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


def save_model(net, optimizer, epoch, identifier):
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, identifier)


def get_tranformer_model_and_encoder(checkpoint):
    MAX_LENGTH = 256
    EMBEDDING_SIZE = 512
    NUM_LAYERS = 6
    model = Transformer(ALPHABET_SIZE, EMBEDDING_SIZE, NUM_LAYERS).eval()
    model = torch.nn.DataParallel(model)
    CHECKPOINT = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
    model.load_state_dict(CHECKPOINT['state_dict'])
    model = model.module.cpu()  # unwrap from nn.DataParallel
    encoder = model.encoder.cpu()

    return model, encoder


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
