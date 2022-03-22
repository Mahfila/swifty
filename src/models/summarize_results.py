import pandas as pd
import argparse
import os
from src.utils.swift_dock_logger import swift_dock_logger
from scipy import stats
from os.path import exists

logger = swift_dock_logger()

parser = argparse.ArgumentParser(description="train code for training a network to estimate depth", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--targets", type=str, help="specify the target protein to ", nargs='+')
parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs='+')
parser.add_argument("--training_sizes", type=int, help="number of epochs", nargs='+')
parser.add_argument("--regressors", type=str, help="specify the training regressors", nargs='+')
args = parser.parse_args()

descriptors_dictionary = {'morgan_onehot_mac_circular': [4755, 'morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile)']}

training_metrics_dir = '../../results/training_metrics/'
testing_metrics_dir = '../../results/testing_metrics/'
test_predictions_dir = '../../results/test_predictions/'
project_info_dir = '../../results/project_info/'
summarized_results_dir = '../../results/summarized_results/'
immediate_results_dir = '../../results/immediate_results/'
os.makedirs(summarized_results_dir, exist_ok=True)
os.makedirs(immediate_results_dir, exist_ok=True)
dataset_dir = "../../datasets"
logger.info(f"Generating Results Has Started")


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
    data_frame.to_csv(f'{immediate_results_dir}immediate_results.csv', index=targets)
    return all_results


if __name__ == '__main__':
    logger.info(f"Generating Results Has Started")
    # summarize_results(args.targets, args.regressors, args.training_sizes)
    logger.info(f"Generating Results Has Ended")
    targets = ['ace', 'spike']
    models = ['decision_tree', 'swift_dock']
    result = get_more_a_specific_result(targets, models, "7000", "morgan_onehot_mac_circular", "test_rsquared")
    print(result)
