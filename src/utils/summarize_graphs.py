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
tanimoto_results_dir = "../../results/tanimoto_results/"
descriptors_dictionary = {'morgan_onehot_mac_circular': [4755, 'morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile)']}

training_metrics_dir = '../../results/training_metrics/'
testing_metrics_dir = '../../results/testing_metrics/'
summarized_results_dir = '../../results/summarized_results/'
immediate_results_dir = '../../results/immediate_results/'
os.makedirs(summarized_results_dir, exist_ok=True)
os.makedirs(immediate_results_dir, exist_ok=True)


def summarize_training_results_group_dataset_two(targets, training_sizes):
    train_r2_list = []
    train_mse_list = []
    train_mae_list = []
    models = ['lstm', 'decision_tree', 'sgdreg', 'xgboost']
    descriptor = 'morgan_onehot_mac'
    labels = []
    for model in models:
        for target in targets:
            for size in training_sizes:
                labels.append(f"{model} - {target} - {size // 1000}k")
                training_results = f"{training_metrics_dir}{model}_{target}_{descriptor}_{size}_train_metrics.csv"
                train_r2 = round(pd.read_csv(training_results)['average_fold_rsquared'].tolist()[0], 2)
                train_mse = round(pd.read_csv(training_results)['average_fold_mse'].tolist()[0], 2)
                train_mae = round(pd.read_csv(training_results)['average_fold_mae'].tolist()[0], 2)
                train_r2_list.append(train_r2)
                train_mse_list.append(train_mse)
                train_mae_list.append(train_mae)
    dict_of_results = {'label': labels,
                       'training average r2': train_r2_list,
                       'training average mse': train_mse_list,
                       'training average mae': train_mae_list}
    result_df = pd.DataFrame(dict_of_results)
    result_df.to_csv(f'{immediate_results_dir}_dataset_two_training_results.csv')


def summarize_training_results_group_dataset_one(targets, training_sizes):
    train_r2_list = []
    train_mse_list = []
    train_mae_list = []
    models = ['lstm', 'decision_tree', 'sgdreg', 'xgboost']
    descriptor = 'morgan_onehot_mac'
    labels = []
    for model in models:
        for target in targets:
            for size in training_sizes:
                target_label = target.replace('1', '_one').replace('2', '_two').replace('3', '_three')
                labels.append(f"{model} - {target_label} - {size // 1000}k")
                training_results = f"{training_metrics_dir}{model}_{target}_{descriptor}_{size}_train_metrics.csv"
                try:
                    train_r2 = round(pd.read_csv(training_results)['train_rsquared'].tolist()[0], 2)

                except:
                    train_r2 = round(pd.read_csv(training_results)['average_fold_rquared'].tolist()[0], 2)
                train_mse = round(pd.read_csv(training_results)['average_fold_mse'].tolist()[0], 2)
                train_mae = round(pd.read_csv(training_results)['average_fold_mae'].tolist()[0], 2)
                train_r2_list.append(train_r2)
                train_mse_list.append(train_mse)
                train_mae_list.append(train_mae)
    dict_of_results = {'label': labels,
                       'training average r2': train_r2_list,
                       'training average mse': train_mse_list,
                       'training average mae': train_mae_list}
    result_df = pd.DataFrame(dict_of_results)
    result_df.to_csv(f'{immediate_results_dir}_dataset_one_training_results.csv')


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
                    target_rounded = [round(item, 2) for item in target_data]
                    predictions_rounded = [round(item, 2) for item in predictions_data]
                    correlation = stats.spearmanr(target_rounded, predictions_rounded)
                    correlation_rounded = round(correlation[0], 2)
                    correlations.append(correlation_rounded)
                    test_r2 = pd.read_csv(testing_results)['test_rsquared'].tolist()[0]
                    test_mae = pd.read_csv(testing_results)['test_mae'].tolist()[0]
                    test_mse = pd.read_csv(testing_results)['test_mse'].tolist()[0]
                    target_list.append(str(size))
                    test_mae_list.append(test_mae)
                    test_mse_list.append(test_mse)
                    test_r2_list.append(test_r2)
                    train_r2_list.append(train_r2)

                df = pd.DataFrame(list(zip(target_list, train_r2_list, test_mae_list, test_mse_list, test_r2_list, correlations)),
                                  columns=['size', 'train_r2', 'test_mae', 'test_mse', 'test_r2', 'Pearson_correlation'])
                df.to_csv(f'{summarized_results_dir}{regressor}_{target}_test_results_summarized.csv', index=False)


def get_a_specific_result(target, training_size, model, metric_type, descriptor):
    file_name = f"{testing_metrics_dir}{model}_{target}_{descriptor}_{training_size}_test_metrics.csv"
    file_exists = exists(file_name)
    if file_exists is False:
        return 0
    if metric_type in ['test_mse', 'test_mae'] and model in ['decision_tree', 'sgdreg', 'xgboost'] and target in ['target1', 'target2', 'target3']:
        result = round(pd.read_csv(file_name)[metric_type].tolist()[0] / 100, 2)
    else:
        result = round(pd.read_csv(file_name)[metric_type].tolist()[0] / 100, 2)
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
    mean = data_frame.mean().tolist()
    rounded_mean = [round(item, 2) for item in mean]
    data_frame.loc['mean'] = rounded_mean
    data_frame.to_csv(f'{immediate_results_dir}{metric_type}_{training_size}_immediate_results.csv', index=targets)
    return all_results


def scatter_plot_predictions_one_dimensional(target_lists, model_lists, descriptor_list, training_size):
    logger.info(f"Generating Scatter Plot 1 dimensional  Has Started")
    fig, ax = plt.subplots(1, max(len(target_lists), len(model_lists)))
    fig.tight_layout(pad=0.4, h_pad=1.7)
    fig.set_size_inches(90, 35)
    fig.set_dpi(150)
    plt.xlim([-9, 0])
    plt.ylim([-8, -2])
    index = 0
    for target in target_lists:
        for model in model_lists:
            color = None
            if index == 1:
                color = 'salmon'
            elif index == 2:
                color = 'orange'
            elif index == 3:
                color = 'purple'
            file_name = f"{test_predictions_dir}{model}_{target}_{descriptor_list}_{training_size}_test_predictions.csv"
            data = pd.read_csv(file_name)
            target_data = data['target']
            predictions_data = data['f1']
            ax[index].scatter(target_data, predictions_data, color=color)
            ax[index].set_title(model, fontsize=50)
            ax[index].set_xlabel('actual docking score', fontsize=45)
            ax[index].set_ylabel('predicted docking score', fontsize=45)
            ax[index].tick_params(axis='x', labelsize=40)
            ax[index].tick_params(axis='y', labelsize=40)
            index = index + 1

    fig.savefig(f"{graph_results_dir}{training_size}_training_plot", facecolor='w')
    logger.info(f"Generating Scatter Plot Has Ended")


def scatter_plot_predictions(target_lists, model_lists, descriptor_list, training_size):
    primary_targets = ['target1', 'target2', 'target3','ace', 'spike', 'nsp', 'nsp_sam']
    original_names = ['Drp1_GTPase', 'RyR2', 'Drp1_MiD49','ace', 'spike', 'nsp', 'nsp_sam']
    target_name_dict = {primary: original for primary, original in zip(primary_targets, original_names)}

    target_dict = {}
    logger.info(f"Generating Scatter Plot Has Started")
    fig, ax = plt.subplots(len(target_lists), len(model_lists),figsize=(100, 85)) #w,h
    font_size = 60
    fig.tight_layout(pad=5, h_pad=27, w_pad=27)
    fig.subplots_adjust(top=0.95, bottom=0.05,left=0.05, right=0.95)  # Adjust top and bottom margins
    fig.set_dpi(200)
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
            name = 'XGBoost'
            if model == 'lstm':
                name = 'LSTM'
            elif model == 'decision_tree':
                name = 'decision-tree'
            elif model == 'sgdreg':
                name = 'SGDR'
            ax[row, col].set_title(f"{name} - {target_name_dict[target]}", fontsize=font_size)
            ax[row, col].set_xlabel('Actual docking score', fontsize=font_size - 5)
            ax[row, col].set_ylabel('Predicted docking score', fontsize=font_size - 5)
            ax[row, col].tick_params(axis='x', labelsize=font_size, rotation=45)  # Add rotation to x-axis labels
            ax[row, col].tick_params(axis='y', labelsize=font_size)
            ax[row, col].set_xlim([-11, 1])  # Adjust x-axis limits
            ax[row, col].set_ylim([-10, -1])  # Adjust y-axis limits

            col = col + 1

        col = 0
        row = row + 1
    fig.savefig(f"{graph_results_dir}{training_size}_training_plot", facecolor='w')
    logger.info(f"Generating Scatter Plot Has Ended")


def correlation_graph(target, title, model_lists, label_list, training_sizes):
    logger.info(f"Correlation Plot Generation Has Started")
    descriptor = 'morgan_onehot_mac'
    correlation_list = []
    for model in model_lists:
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

    x = np.arange(len(label_list))  # the label locations
    width = 0.2  # the width of the bars
    bar1 = np.arange(len(label_list))  # the label locations
    bar2 = [i + width for i in bar1]
    bar3 = [i + width for i in bar2]
    bar4 = [i + width for i in bar3]
    fig, ax = plt.subplots(1, 1)
    fig.set_dpi(300)
    model_names = ['LSTM', 'decision-tree', 'SGDR', 'XGBoost']
    rects1 = ax.bar(bar1, correlation_list[0], width, color='orange', label=model_names[0])
    rects2 = ax.bar(bar2, correlation_list[1], width, color='blue', label=model_names[1])
    rects3 = ax.bar(bar3, correlation_list[2], width, color='green', label=model_names[2])
    rects4 = ax.bar(bar4, correlation_list[3], width, color='salmon', label=model_names[3])

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Correlation score')
    ax.set_xlabel('Training set size')
    ax.set_xticks(x, label_list)
    ax.set_ylim(0, 1)
    ax.legend(fancybox=True, framealpha=1, loc='lower left', shadow=True, borderpad=1)
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)
    ax.bar_label(rects4, padding=3)
    plt.title(title)

    fig.tight_layout()
    plt.show()
    fig.savefig(f"{graph_results_dir}{target}_correlation_plot", facecolor='w')
    logger.info(f"Correlation Plot Generation Has Ended")


def average_correlation(model_list, training_sizes, target):
    descriptor = 'morgan_onehot_mac'
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


def creating_training_time(training_sizes, models, target):
    def add_labels(y):
        for i in range(len(y)):
            plt.text(i, y[i], y[i], ha='center', fontsize=10)

    labels_for_times = []
    for model in models:
        for training_size in training_sizes:
            labels_for_times.append(f"{model} - {training_size // 1000}k")

    descriptor = 'morgan_onehot_mac'
    all_test_list = []
    all_train_list = []
    for model in models:
        for train_size in training_sizes:
            file_name = f"{project_info_dir}{model}_{target}_{descriptor}_{train_size}_project_info.csv"
            logger.info(f"Current File -- {file_name}")
            data = pd.read_csv(file_name)
            train_time = data['5 fold_validation_time'].tolist()[0]
            test_time = data['testing_time'].tolist()[0]
            train_time_rounded = round(train_time / 5, 1)
            test_time_rounded = round(test_time / 5, 1)
            all_train_list.extend([train_time_rounded])
            all_test_list.extend([test_time_rounded])

    summation_of_train_and_test_times = []
    for a, b in zip(all_test_list, all_train_list):
        summation_of_train_and_test_times.append(round(a + b, 1))

    all_train_list_new = []
    all_test_list_new = []
    for item in all_train_list:
        if item <= 1:
            all_train_list_new.append('< 1')
        else:
            all_train_list_new.append(str(item))

    for item in all_test_list:
        if item <= 1:
            all_test_list_new.append('< 1')
        else:
            all_test_list_new.append(str(item))
    dict_of_results = {'label': labels_for_times,
                       'training time': all_train_list_new,
                       'testing time': all_test_list_new,
                       'total time': summation_of_train_and_test_times}
    result_df = pd.DataFrame(dict_of_results)
    result_df.to_csv(f'{immediate_results_dir}{target}_times.csv')

    # men_means = all_test_list
    # women_means = all_train_list
    # width = 0.70
    # fig, ax = plt.subplots()
    # fig.set_dpi(300)
    # ax.bar(labels_for_times, men_means, width, label='Testing Time')
    # ax.bar(labels_for_times, women_means, width, bottom=men_means, label='Training Time')
    # add_labels(summation_of_train_and_test_times)
    # ax.set_ylabel('Time in CPU (Minutes)')
    # plt.xticks(rotation=70)
    # plt.savefig(f"{graph_results_dir}{target}_times", facecolor='w', dpi=200, bbox_inches='tight')


def ven_diagram(target, models, training_size, descriptor, tok_k):
    num_models = len(models)
    fig, axes = plt.subplots(nrows=1, ncols=num_models, figsize=(5 * num_models, 5))

    for ax, model in zip(axes, models):
        file_name = f"{test_predictions_dir}{model}_{target}_{descriptor}_{training_size}_test_predictions.csv"
        logger.info(f"Current File -- {file_name}")
        data = pd.read_csv(file_name)
        ids = [f"l{i}" for i in range(len(data))]
        data['ids'] = ids
        target_data = data[['target', 'ids']].astype({"target": float})
        target_data['target'] = target_data['target'].round(2)
        prediction = data[['f1', 'ids']].astype({"f1": float})
        prediction['f1'] = prediction['f1'].round(2)
        prediction = prediction.sort_values('f1')[0:tok_k]
        target_data = target_data.sort_values('target')[0:tok_k]
        prediction_set = set(prediction['ids'].tolist())
        target_set = set(target_data['ids'].tolist())
        union_len = len(prediction_set.intersection(target_set))
        pred_len = len(prediction_set.difference(target_set))
        tar_len = len(target_set.intersection(prediction_set))

        v = venn2(subsets=(tar_len, pred_len, union_len), set_labels=['Target', 'Prediction'], ax=ax,
                  set_colors=('blue', 'red'), alpha=0.5)
        for text in v.set_labels:
            text.set_fontsize(20)
        c = venn2_circles(subsets=(tar_len, pred_len, union_len), linestyle='-', linewidth=2, color="black", ax=ax)

        name = 'XGBoost'
        if model == 'lstm':
            name = 'LSTM'
        elif model == 'decision_tree':
            name = 'Decision Tree'
        elif model == 'sgdreg':
            name = 'SGDR'

        ax.set_title(f"top 10k using {name}", fontsize=20)

    plt.tight_layout()
    plt.savefig(f"{graph_results_dir}{target}_ven_diagrams.png", facecolor='w')
    plt.clf()


def ven_diagram_for_single_target(targets, model, training_size, descriptor, tok_k):
    primary_targets = ['target1', 'target2', 'target3', 'ace', 'spike', 'nsp', 'nsp_sam']
    original_names = ['Drp1_GTPase', 'RyR2', 'Drp1_MiD49', 'ace', 'spike', 'nsp', 'nsp_sam']
    target_name_dict = {primary: original for primary, original in zip(primary_targets, original_names)}
    num_columns = 4
    num_rows = (len(targets) + num_columns - 1) // num_columns

    fig, axes = plt.subplots(nrows=num_rows, ncols=num_columns, figsize=(5 * num_columns, 5 * num_rows))

    for idx, target in enumerate(targets):
        current_row = idx // num_columns
        current_col = idx % num_columns

        file_name = f"{test_predictions_dir}{model}_{target}_{descriptor}_{training_size}_test_predictions.csv"
        logger.info(f"Current File -- {file_name}")
        data = pd.read_csv(file_name)
        ids = [f"l{i}" for i in range(len(data))]
        data['ids'] = ids
        target_data = data[['target', 'ids']].astype({"target": float})
        target_data['target'] = target_data['target'].round(2)
        prediction = data[['f1', 'ids']].astype({"f1": float})
        prediction['f1'] = prediction['f1'].round(2)
        prediction = prediction.sort_values('f1')[0:tok_k]
        target_data = target_data.sort_values('target')[0:tok_k]
        prediction_set = set(prediction['ids'].tolist())
        target_set = set(target_data['ids'].tolist())
        union_len = round((len(prediction_set.intersection(target_set))/tok_k) *100,2)
        pred_len = round((len(prediction_set.difference(target_set))/tok_k) *100,2)
        tar_len = round((len(target_set.intersection(prediction_set)) /tok_k) *100,2)

        v = venn2(subsets=(tar_len, pred_len, union_len), set_labels=['Target', 'Prediction'], set_colors=('blue', 'red'), alpha=0.5, ax=axes[current_row, current_col])
        for text in v.set_labels:
            text.set_fontsize(20)
        c = venn2_circles(subsets=(tar_len, pred_len, union_len), linestyle='-', linewidth=2, color="black", ax=axes[current_row, current_col])
        name = 'XGBoost'
        if model == 'lstm':
            name = 'LSTM'
        elif model == 'decision_tree':
            name = 'decision-tree'
        elif model == 'sgdreg':
            name = 'SGDR'
        axes[current_row, current_col].set_title(f"{target_name_dict[target]}", fontsize=20)

    # Remove unused subplots
    for idx in range(len(targets), num_rows * num_columns):
        current_row = idx // num_columns
        current_col = idx % num_columns
        fig.delaxes(axes[current_row][current_col])

    plt.tight_layout()
    plt.savefig(f"{graph_results_dir}{model}_ven_diagrams.png", facecolor='w')
    plt.clf()

def get_project_info(target):
    descriptor = 'morgan_onehot_mac_circular'
    model = 'decision_tree'
    training_sizes = [7000, 10000, 20000, 50000]
    data_file = f"{dataset_dir}/{target}.csv"
    data = pd.read_csv(data_file)
    data = data.dropna()
    print("data size = ", len(data))
    for train_size in training_sizes:
        file_name = f"{project_info_dir}{model}_{target}_{descriptor}_{train_size}_project_info.csv"
        logger.info(f"Current File -- {file_name}")
        data = pd.read_csv(file_name)
        test_time = data['testing_size'].tolist()[0]
        print(f"training = {train_size}, testing size = {test_time}")


def plot_tanimoto_distances_version_two(target_one, target_two):
    tanimoto_stats_one = pd.read_csv(f"{tanimoto_results_dir}/{target_one}/all_distances.csv")
    tanimoto_stats_two = pd.read_csv(f"{tanimoto_results_dir}/{target_two}/all_distances.csv").sample(len(tanimoto_stats_one))
    avg_distances_one = tanimoto_stats_one['avg_distances']
    max_distances_one = tanimoto_stats_one['max_distances']
    min_distances_one = tanimoto_stats_one['min_distances']
    avg_distances_two = tanimoto_stats_two['avg_distances']
    max_distances_two = tanimoto_stats_two['max_distances']
    min_distances_two = tanimoto_stats_two['min_distances']
    font_size = 160
    plot_height = 200
    plot_width = 140
    bins = 20

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(plot_height, plot_width))
    plt.subplots_adjust(hspace=0.3)
    ax1.hist(avg_distances_one, alpha=0.5, color="skyblue", label="spike")
    ax1.hist(avg_distances_two, alpha=0.5, color="salmon", bins=bins, label="Drp1_GTPase")
    ax1.legend(loc="upper left", prop={'size': font_size})
    ax1.set_xlabel('Fingerprints average similarity', fontsize=font_size)
    ax1.set_ylabel('Frequency', fontsize=font_size)
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.tick_params(axis='y', labelsize=font_size)

    ax2.hist(max_distances_one, alpha=0.5, color="skyblue", bins=bins, label="spike")
    ax2.hist(max_distances_two, alpha=0.5, color="salmon", bins=bins, label="Drp1_GTPase")
    ax2.legend(loc='upper left', prop={'size': font_size})
    ax2.set_xlabel('Fingerprints maximum similarity', fontsize=font_size)
    ax2.set_ylabel('Frequency', fontsize=font_size)
    ax2.tick_params(axis='x', labelsize=font_size)
    ax2.tick_params(axis='y', labelsize=font_size)

    ax3.hist(min_distances_one, alpha=0.5, color="skyblue", bins=bins, label="spike")
    ax3.hist(min_distances_two, alpha=0.5, color="salmon", bins=bins, label="Drp1_GTPase")
    ax3.legend(loc='upper right', prop={'size': font_size})
    ax3.set_xlabel('Fingerprints minimum similarity', fontsize=font_size)
    ax3.set_ylabel('Frequency', fontsize=font_size)

    ax3.tick_params(axis='x', labelsize=font_size)
    ax3.tick_params(axis='y', labelsize=font_size)
    fig.savefig(f"{graph_results_dir}_tanimoto", facecolor='w')


def datasets_histogram():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    plt.subplots_adjust(wspace=0.4)
    fig.set_dpi(300)
    fig.set_size_inches(60, 15)
    bin_size = 30
    font_size = 50
    targets = ['target1', 'target2', 'target3']
    data1 = pd.read_csv(f'/Users/abdulsalamyazid/PycharmProjects/swift_dock/datasets/{targets[0]}.csv')
    data2 = pd.read_csv(f'/Users/abdulsalamyazid/PycharmProjects/swift_dock/datasets/{targets[1]}.csv')
    data3 = pd.read_csv(f'/Users/abdulsalamyazid/PycharmProjects/swift_dock/datasets/{targets[2]}.csv')

    data1_scores = data1['docking_score'].tolist()
    data2_scores = data2['docking_score'].tolist()
    data3_scores = data3['docking_score'].tolist()
    ax1.hist(data1_scores, bins=bin_size)
    ax1.set_title('target one', fontsize=font_size)
    ax1.set_xlabel('Docking Score', fontsize=font_size)
    ax1.set_ylabel('Frequency', fontsize=font_size)
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.tick_params(axis='y', labelsize=font_size)

    # Plot 2
    ax2.hist(data2_scores, color="orange", bins=bin_size)
    ax2.set_title('target two', fontsize=font_size)
    ax2.set_xlabel('Docking Score', fontsize=font_size)
    ax2.set_ylabel('Frequency', fontsize=font_size)
    ax2.tick_params(axis='x', labelsize=font_size)
    ax2.tick_params(axis='y', labelsize=font_size)

    # Plot 3
    ax3.hist(data3_scores, color="green", bins=bin_size)
    ax3.set_title('target three', fontsize=font_size)
    ax3.set_xlabel('Docking Score', fontsize=font_size)
    ax3.set_ylabel('Frequency', fontsize=font_size)
    ax3.tick_params(axis='x', labelsize=font_size)
    ax3.tick_params(axis='y', labelsize=font_size)
    fig.savefig(f"{graph_results_dir}hist_data_one", facecolor='w')

    targets = ['nsp', 'nsp_sam', 'spike', 'ace']
    data1 = pd.read_csv(f'/Users/abdulsalamyazid/PycharmProjects/swift_dock/datasets/{targets[0]}.csv')
    data2 = pd.read_csv(f'/Users/abdulsalamyazid/PycharmProjects/swift_dock/datasets/{targets[1]}.csv')
    data3 = pd.read_csv(f'/Users/abdulsalamyazid/PycharmProjects/swift_dock/datasets/{targets[2]}.csv')
    data4 = pd.read_csv(f'/Users/abdulsalamyazid/PycharmProjects/swift_dock/datasets/{targets[3]}.csv')

    data1_scores = data1['docking_score'].tolist()
    data2_scores = data2['docking_score'].tolist()
    data3_scores = data3['docking_score'].tolist()
    data4_scores = data4['docking_score'].tolist()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
    plt.subplots_adjust(wspace=0.4)
    fig.set_size_inches(60, 15)
    # Plot 1
    ax1.hist(data1_scores, bins=bin_size)
    ax1.set_title(targets[0], fontsize=font_size)
    ax1.set_xlabel('Docking Score', fontsize=font_size)
    ax1.set_ylabel('Frequency', fontsize=font_size)
    ax1.tick_params(axis='x', labelsize=font_size)
    ax1.tick_params(axis='y', labelsize=font_size)

    # Plot 2
    ax2.hist(data2_scores, color="orange", bins=bin_size)
    ax2.set_title('nsp-sam', fontsize=font_size)
    ax2.set_xlabel('Docking Score', fontsize=font_size)
    ax2.set_ylabel('Frequency', fontsize=font_size)
    ax2.tick_params(axis='x', labelsize=font_size)
    ax2.tick_params(axis='y', labelsize=font_size)

    # Plot 3
    ax3.hist(data3_scores, color="green", bins=bin_size)
    ax3.set_title(targets[2], fontsize=font_size)
    ax3.set_xlabel('Docking Score', fontsize=font_size)
    ax3.set_ylabel('Frequency', fontsize=font_size)
    ax3.tick_params(axis='x', labelsize=font_size)
    ax3.tick_params(axis='y', labelsize=font_size)

    ax4.hist(data4_scores, color="purple", bins=bin_size)
    ax4.set_title(targets[3], fontsize=font_size)
    ax4.set_xlabel('Docking Score', fontsize=font_size)
    ax4.set_ylabel('Frequency', fontsize=font_size)
    ax4.tick_params(axis='x', labelsize=font_size)
    ax4.tick_params(axis='y', labelsize=font_size)

    fig.savefig(f"{graph_results_dir}hist_data_two", facecolor='w')
