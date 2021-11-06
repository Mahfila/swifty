from torch.utils.data import DataLoader
from data_generator import DataGenerator
from utils import test_model, create_fold_predictions_and_target_df, create_test_metrics


def test_models(test, number_of_folds, all_networks, descriptor, TESTING_SIZE):
    all_models_predictions = []

    smiles_data_test = DataGenerator(test, descriptor=descriptor)
    test_dataloader = DataLoader(smiles_data_test, batch_size=128, shuffle=False, num_workers=28)

    for fold in range(number_of_folds):
        print("making fold ", fold, " predictions")
        test_predictions = test_model(test_dataloader, all_networks[fold])
        all_models_predictions.append(test_predictions)

    smiles_target = test['docking_score'].tolist()
    metrics_dict_test = create_test_metrics(all_models_predictions, smiles_target, number_of_folds, TESTING_SIZE)
    predictions_and_target_df = create_fold_predictions_and_target_df(all_models_predictions, smiles_target, number_of_folds, TESTING_SIZE)
    return metrics_dict_test, predictions_and_target_df
