import os
import matplotlib.pyplot as plt
import torch
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error


def calculate_metrics(predictions, target):
    mse = mean_squared_error(target, predictions)
    mae = mean_absolute_error(target, predictions)
    rsquared = r2_score(target, predictions)
    return mse, mae, rsquared


def get_training_and_test_data(DATA, TRAINING_SIZE, TESTING_SIZE=3000000):
    DATA = DATA.dropna()
    del DATA["Unnamed: 0"]
    train = DATA.sample(TRAINING_SIZE)
    test = pd.concat([train, DATA]).drop_duplicates(keep=False)
    test = test.sample(TESTING_SIZE)
    return train, test

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
