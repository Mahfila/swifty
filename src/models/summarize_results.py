import pandas as pd
import argparse
import os
from src.utils.swift_dock_logger import swift_dock_logger
from scipy import stats

logger = swift_dock_logger()

parser = argparse.ArgumentParser(description="train code for training a network to estimate depth", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--targets", type=str, help="specify the target protein to ", nargs='+')
parser.add_argument("--descriptors", type=str, help="specify the training descriptor", nargs='+')
parser.add_argument("--training_sizes", type=int, help="number of epochs", nargs='+')
parser.add_argument("--regressors", type=str, help="specify the training regressors", nargs='+')
args = parser.parse_args()

#####Swift Dock Arugments

descriptors_dictionary = {'morgan_onehot_mac_circular': [4755, 'morgan_fingerprints_mac_and_one_hot_descriptors_CircularFingerprint(smile)']}

training_metrics_dir = '../../results/training_metrics/'
testing_metrics_dir = '../../results/testing_metrics/'
test_predictions_dir = '../../results/test_predictions/'
project_info_dir = '../../results/project_info/'
summarized_results_dir = '../../results/summarized_results/'
os.makedirs(summarized_results_dir, exist_ok=True)

dataset_dir = "../../datasets"

logger.info(f"Generating Results Has Started")
if __name__ == '__main__':
    for target in args.targets:
        for regressor in args.regressors:
            for key, value in descriptors_dictionary.items():
                target_list = []
                test_mse_list = []
                test_mae_list = []
                test_r2_list = []
                train_r2_list = []
                correlations = []
                for size in args.training_sizes:
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

logger.info(f"Generating Results Has Ended")
