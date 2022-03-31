import pandas as pd
import os
import numpy as np
from src.utils.swift_dock_logger import swift_dock_logger
from scipy import stats
import matplotlib.pyplot as plt
from os.path import exists
from matplotlib_venn import venn2, venn2_circles

logger = swift_dock_logger()

graph_results_dir = '../../results/graph_results/'
test_predictions_dir = '../../results/test_predictions/'
project_info_dir = '../../results/project_info/'
os.makedirs(graph_results_dir, exist_ok=True)
dataset_dir = "../../datasets"
descriptors_dictionary = {'morgan_onehot_mac_circular': [4755, 'morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile)']}

training_metrics_dir = '../../results/training_metrics/'
testing_metrics_dir = '../../results/testing_metrics/'
summarized_results_dir = '../../results/summarized_results/'
immediate_results_dir = '../../results/immediate_results/'
os.makedirs(summarized_results_dir, exist_ok=True)
os.makedirs(immediate_results_dir, exist_ok=True)


def summarize_results(targets, regressors, training_sizes):
    for target in targets:
        for regressor in regressors:
            for key, value in descriptors_dictionary.items():
                target_list = []
                test_mse_list = []
                test_mae_list = []
                test_r2_list = []
                train_r2_list = []
                correlations = []
                for size in training_sizes:
                    training_results = f"{training_metrics_dir}{regressor}_{target}_{key}_{str(size)}_train_metrics.csv"
                    try:
                        train_r2 = round(pd.read_csv(training_results)['train_rsquared'].tolist()[0], 2)
                    except:
                        train_r2 = round(pd.read_csv(training_results)['average_fold_rsquared'].tolist()[0], 2)
                    testing_results = f"{testing_metrics_dir}{regressor}_{target}_{key}_{str(size)}_test_metrics.csv"
                    predictions = pd.read_csv(f"{test_predictions_dir}{regressor}_{target}_{key}_{str(size)}_test_predictions.csv")
                    target_data = predictions['target'].tolist()
                    predictions_data = predictions['f1'].tolist()
                    target_top_5000_rounded = [round(item, 2) for item in target_data]
                    predictions_top_5000_rounded = [round(item, 2) for item in predictions_data]
                    correlation = stats.spearmanr(target_top_5000_rounded, predictions_top_5000_rounded)
                    correlation_rounded = round(correlation[0], 2)
                    correlations.append(correlation_rounded)
                    test_r2 = round(pd.read_csv(testing_results)['test_rsquared'].tolist()[0], 2)
                    test_mae = round(pd.read_csv(testing_results)['test_mae'].tolist()[0], 2)
                    test_mse = round(pd.read_csv(testing_results)['test_mse'].tolist()[0], 2)
                    target_list.append(str(size))
                    test_mae_list.append(test_mae)
                    test_mse_list.append(test_mse)
                    test_r2_list.append(test_r2)
                    train_r2_list.append(train_r2)

                df = pd.DataFrame(list(zip(target_list, train_r2_list, test_mae_list, test_mse_list, test_r2_list, correlations)),
                                  columns=['size', 'train_r2', 'test_mae', 'test_mse', 'test_r2', 'Pearson_correlation'])
                df.to_csv(f'{summarized_results_dir}{regressor}_{target}.csv', index=False)


def get_a_specific_result(target, training_size, model, metric_type, descriptor):
    file_name = f"{testing_metrics_dir}{model}_{target}_{descriptor}_{training_size}_test_metrics.csv"
    file_exists = exists(file_name)
    if file_exists is False:
        return 0
    result = round(pd.read_csv(file_name)[metric_type].tolist()[0], 2)
    return result


def get_more_a_specific_result(targets, models, training_size, descriptor, metric_type):
    all_results = []
    for target in targets:
        result_list = []
        for model in models:
            result_list.append(get_a_specific_result(target, training_size, model, metric_type, descriptor))
        all_results.append(result_list)
    data_frame = pd.DataFrame(all_results, columns=models, index=targets)
    data_frame.index.name = 'target'
    data_frame.to_csv(f'{immediate_results_dir}{metric_type}_{training_size}_immediate_results.csv', index=targets)
    return all_results


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


def correlation_graph(target_lists, model_lists, label_list, training_sizes):
    logger.info(f"Correlation Plot Generation Has Started")
    descriptor = 'morgan_onehot_mac_circular'
    correlation_list = []
    for model in model_lists:
        each_train_size_list = []
        for train_size in training_sizes:
            file_name = f"{test_predictions_dir}{model}_{target_lists}_{descriptor}_{train_size}_test_predictions.csv"
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

    x = np.arange(len(label_list))  # the label locations
    width = 0.2  # the width of the bars
    bar1 = np.arange(len(label_list))  # the label locations
    bar2 = [i + width for i in bar1]
    bar3 = [i + width for i in bar2]
    bar4 = [i + width for i in bar3]

    fig, ax = plt.subplots(1, 1)
    fig.set_dpi(200)

    rects1 = ax.bar(bar1, correlation_list[0], width, color='orange', label=model_lists[0])
    rects2 = ax.bar(bar2, correlation_list[1], width, color='blue', label=model_lists[1])
    rects3 = ax.bar(bar3, correlation_list[2], width, color='green', label=model_lists[2])
    rects4 = ax.bar(bar4, correlation_list[3], width, color='salmon', label=model_lists[3])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Correlation Score')
    ax.set_xticks(x, label_list)
    ax.set_ylim(0, 1)
    ax.legend(fancybox=True, framealpha=1, loc='lower left', shadow=True, borderpad=1)
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)

    fig.tight_layout()
    plt.show()
    fig.savefig(f"{graph_results_dir}{target_lists}_correlation_plot", facecolor='w')
    logger.info(f"Correlation Plot Generation Has Ended")


def average_correlation(model_list, training_sizes, target):
    descriptor = 'morgan_onehot_mac_circular'
    correlation_list = []
    for model in model_list:
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


def creating_training_time(training_sizes, labels_for_times, models, targets):
    def add_labels(y):
        for i in range(len(y)):
            plt.text(i, y[i], y[i], ha='center', fontsize=10)

    metric = 'morgan_onehot_mac_circular'
    all_test_list = []
    all_train_list = []
    for model in models:
        for train_size in training_sizes:
            for target in targets:
                file_name = f"{project_info_dir}{model}_{target}_{metric}_{train_size}_project_info.csv"
                logger.info(f"Current File -- {file_name}")
                data = pd.read_csv(file_name)
                train_time = data['5 fold_validation_time'].tolist()[0]
                test_time = data['testing_time'].tolist()[0]
                train_time_rounded = round(train_time, 1) // 5
                test_time_rounded = round(test_time, 1) // 5
            all_train_list.extend([train_time_rounded])
            all_test_list.extend([test_time_rounded])

    summation_of_train_and_test_times = []
    for a, b in zip(all_test_list, all_train_list):
        summation_of_train_and_test_times.append(a + b)

    men_means = all_test_list
    women_means = all_train_list
    width = 0.70
    fig, ax = plt.subplots()
    fig.set_dpi(300)
    ax.bar(labels_for_times, men_means, width, label='Testing Time')
    ax.bar(labels_for_times, women_means, width, bottom=men_means, label='Training Time')
    add_labels(summation_of_train_and_test_times)
    ax.set_ylabel('Time in CPU Minues')
    plt.xticks(rotation=70)
    plt.savefig(f"{graph_results_dir}times", facecolor='w', dpi=200, bbox_inches='tight')


def ven_diagram(targets, models, training_size, descriptor, tok_k):
    for model in models:
        file_name = f"{test_predictions_dir}{model}_{targets[0]}_{descriptor}_{training_size}_test_predictions.csv"
        logger.info(f"Current File -- {file_name}")
        data = pd.read_csv(file_name)
        ids = [f"l{i}" for i in range(len(data))]
        data['ids'] = ids
        target_data = data[['target', 'ids']].astype({"target": float})
        target_data['target'] = target_data['target'].round(2)
        prediction = data[['f2', 'ids']].astype({"f2": float})
        prediction['f2'] = prediction['f2'].round(2)
        prediction = prediction.sort_values('f2')[0:tok_k]
        target_data = target_data.sort_values('target')[0:tok_k]
        prediction_set = set(prediction['ids'].tolist())
        target_set = set(target_data['ids'].tolist())
        union_len = len(prediction_set.intersection(target_set))
        pred_len = len(prediction_set.difference(target_set))
        tar_len = len(target_set.intersection(prediction_set))
        # if model == 'lstm':
        #     tar_len = 4001
        #     pred_len = 10000 - 4001
        #     union_len = 4001
        v = venn2(subsets=(tar_len, pred_len, union_len), set_labels=['Target', 'Prediction'])
        for text in v.set_labels:
            text.set_fontsize(20)
        c = venn2_circles(subsets=(tar_len, pred_len, union_len), linestyle='-', linewidth=2, color="black")
        plt.title(f"top {tok_k} using {model}{str(training_size).replace('0', '') + 'k'}", fontsize=20)
        plt.savefig(f"{graph_results_dir}{model}_ven diagrams", facecolor='w')
        plt.clf()
