import pandas as pd
import os
import numpy as np
from src.utils.swift_dock_logger import swift_dock_logger
from scipy import stats
import matplotlib.pyplot as plt
from os.path import exists

logger = swift_dock_logger()

graph_results_dir = '../../results/graph_results/'
test_predictions_dir = '../../results/test_predictions/'
os.makedirs(graph_results_dir, exist_ok=True)
dataset_dir = "../../datasets"


def scatter_plot_predictions(target_lists, model_lists, descriptor_list, training_size):
    logger.info(f"Generating Scatter Plot Has Started")
    fig, ax = plt.subplots(len(target_lists), len(model_lists))
    fig.tight_layout(pad=0.4, h_pad=1.7)
    fig.set_size_inches(70, 30)
    fig.set_dpi(30)
    plt.xlim([-9, 0])
    plt.ylim([-8, -2])
    col = 0
    row = 0
    for target in target_lists:
        for model in model_lists:
            color = None
            if col == 1:
                color = 'salmon'
            elif col == 2:
                color = 'orange'
            elif col == 3:
                color = 'purple'
            file_name = f"{test_predictions_dir}{model}_{target}_{descriptor_list}_{training_size}_test_predictions.csv"
            data = pd.read_csv(file_name)
            target_data = data['target']
            predictions_data = data['f1']
            ax[row, col].scatter(target_data, predictions_data, color=color)
            ax[row, col].set_title(model, fontsize=50)
            ax[row, col].set_xlabel('actual docking score', fontsize=45)
            ax[row, col].set_ylabel('predicted docking score', fontsize=45)
            ax[row, col].tick_params(axis='x', labelsize=40)
            ax[row, col].tick_params(axis='y', labelsize=40)
            col = col + 1

        col = 0
        row = row + 1
    fig.savefig(f"{graph_results_dir}{training_size}_training_plot", facecolor='w')
    logger.info(f"Generating Scatter Plot Has Ended")


def correlation_graph(target, models, labels, training_sizes):
    logger.info(f"Correlation Plot Generation Has Started")
    descriptor = 'morgan_onehot_mac_circular'
    correlation_list = []
    for model in models:
        each_train_size_list = []
        for train_size in training_sizes:
            file_name = f"{test_predictions_dir}{model}_{target}_{descriptor}_{train_size}_test_predictions.csv"
            file_exists = exists(file_name)
            if file_exists is False:
                logger.info(f"File not found -- {file_name}")
                return print(FileNotFoundError)
            logger.info(f"Current File {file_name}")
            data = pd.read_csv(file_name)
            target_data = data['target'].tolist()
            predictions_data = data['f1'].tolist()
            target_top_5000_rounded = [round(item, 2) for item in target_data]
            predictions_top_5000_rounded = [round(item, 2) for item in predictions_data]
            correlation = stats.spearmanr(target_top_5000_rounded, predictions_top_5000_rounded)
            correlation_rounded = round(correlation[0], 2)
            each_train_size_list.append(correlation_rounded)
        correlation_list.append(each_train_size_list)

    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    bar1 = np.arange(len(labels))  # the label locations
    bar2 = [i + width for i in bar1]
    bar3 = [i + width for i in bar2]
    bar4 = [i + width for i in bar3]

    fig, ax = plt.subplots(1, 1)
    fig.set_dpi(200)

    rects1 = ax.bar(bar1, correlation_list[0], width, color='orange', label=models[0])
    rects2 = ax.bar(bar2, correlation_list[1], width, color='blue', label=models[1])
    rects3 = ax.bar(bar3, correlation_list[2], width, color='green', label=models[2])
    rects4 = ax.bar(bar4, correlation_list[3], width, color='salmon', label=models[3])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Correlation Score')
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1)
    ax.legend(fancybox=True, framealpha=1, loc='lower left', shadow=True, borderpad=1)
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)

    fig.tight_layout()
    plt.show()
    fig.savefig(f"{graph_results_dir}{target}_correlation_plot", facecolor='w')
    logger.info(f"Correlation Plot Generation Has Ended")


def average_correlation(models, training_sizes, target):

    descriptor = 'morgan_onehot_mac_circular'
    correlation_list = []
    for model in models:
        each_train_size_list = []
        for train_size in training_sizes:
            file_name = f"{test_predictions_dir}{model}_{target}_{descriptor}_{train_size}_test_predictions.csv"
            file_exists = exists(file_name)
            if file_exists is False:
                logger.info(f"File not found -- {file_name}")
                return print(FileNotFoundError)
            logger.info(f"Current File -- {file_name}")
            data = pd.read_csv(file_name)
            target_data = data['target'].tolist()
            predictions_data = data['f1'].tolist()
            target_top_5000_rounded = [round(item, 2) for item in target_data]
            predictions_top_5000_rounded = [round(item, 2) for item in predictions_data]
            correlation = stats.spearmanr(target_top_5000_rounded, predictions_top_5000_rounded)
            correlation_rounded = round(correlation[0], 2)
            each_train_size_list.append(correlation_rounded)
        correlation_list.append(each_train_size_list)

    averages = [sum(item) / len(item) for item in correlation_list]
    return averages


def average_correlations(models, training_sizes, targets):
    logger.info(f"Correlation Table Generation Has Started")
    results = []
    for target in targets:
        results.append({target: average_correlation(models, training_sizes, target)})
    list_of_rows = []
    for item in results:
        key = list(item.keys())[0]
        values = item[key]
        list_of_rows.append(values)
    result_df = pd.DataFrame(list_of_rows, columns=models, index=targets)
    result_df.index.name = 'target'
    result_df.to_csv(f"{graph_results_dir}cor_table.csv")
    logger.info(f"Correlation Table Generation Has Ended")


if __name__ == '__main__':

    target = 'ace'
    targets = ['ace', 'spike','']
    primary_targets = ['target1', 'target2', 'target3']
    models = ['lstm', 'decision_tree', 'sgdreg', 'xgboost']
    descriptor = 'morgan_onehot_mac_circular'
    labels = ['7k', '50k', '350k']
    training_sizes = [7000, 50000, 350000]
    # scatter_plot_predictions(targets, models, descriptor, 7000)
    # correlation_graph('ace', models, labels, training_sizes)
    average_correlations(models, training_sizes, targets)
