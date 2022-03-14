import os
from rdkit import Chem
from torch.utils.data import DataLoader
import pandas as pd
import time
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
from src.utils.utils import TanimotoDataGenerator, save_dict, save_dict_with_one_index
from tqdm.notebook import tqdm_notebook
from tqdm import tqdm

mpl.rcParams['figure.dpi'] = 100
warnings.filterwarnings("ignore")

dataset_dir = '../../datasets/'
result_dir = '../../results/tanimoto_results/'
os.makedirs(result_dir, exist_ok=True)


def calculate_tanimoto(target_name):
    target_dir = f"{result_dir}{target_name}/"
    os.makedirs(target_dir, exist_ok=True)
    original = pd.read_csv(f"{dataset_dir}{target_name}.csv")
    data_len = len(original) * 0.35
    data = original.sample(int(data_len))
    data_docking_scores = data['docking_score'].to_list()
    data = data['smile'].to_dict()
    data_information = {"size of original data": len(original), "size of sampled data": len(data)}
    rdkit_info = {}
    counter = 0

    for key, value in data.items():
        try:
            rdkit_info[counter] = [value, Chem.RDKFingerprint(Chem.MolFromSmiles(value))]
            counter = counter + 1
        except:
            continue

    smiles_data_train = TanimotoDataGenerator(rdkit_info)
    print("Finished 1")
    train_dataloader = DataLoader(smiles_data_train,
                                  batch_size=32,
                                  shuffle=False,
                                  num_workers=16)

    start_time = time.time()
    all_distances = []
    all_maxes = []
    all_mins = []
    for iteration, sample_batched in enumerate(tqdm(train_dataloader)):
        all_distances.extend(sample_batched['avg'].tolist())
        all_maxes.extend(sample_batched['max'].tolist())
        all_mins.extend(sample_batched['min'].tolist())

    result_dict = {"sampled__data_docking_scores": data_docking_scores,
                   "avg_distances": all_distances,
                   "max_distances": all_maxes,
                   "min_distances": all_mins}
    hours = (time.time() - start_time) // 60 // 24
    data_information["time(hr)"] = hours
    result_dir_plots = f"{target_dir}tanimoto_calculation_information.png"
    result_dir_distances = f"{target_dir}all_distances.csv"
    result_dir_extra_information = f"{target_dir}tanimoto_calculation_information.csv"

    save_dict(result_dict, result_dir_distances)
    save_dict_with_one_index(data_information, result_dir_extra_information)

    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(40, 8))
    # fig.set_size_inches(40, 10)

    # Plot 1
    ax1.hist(original['docking_score'])
    ax1.set_title('Original Data', fontsize=30)
    ax1.set_xlabel('Docking Score', fontsize=25)
    ax1.set_ylabel('Frequency', fontsize=25)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    # Plot 2
    ax2.hist(original['docking_score'])
    ax2.set_title('Sampled Data', fontsize=30)
    ax2.set_xlabel('Docking Score', fontsize=25)
    ax2.set_ylabel('Frequency', fontsize=25)
    ax2.tick_params(axis='x', labelsize=20)
    ax2.tick_params(axis='y', labelsize=20)
    # Plot 3
    ax3.hist(all_distances)
    ax3.set_xlabel('Average Tanimoto Distance', fontsize=25)
    ax3.set_ylabel('Frequency', fontsize=25)
    ax3.tick_params(axis='x', labelsize=20)
    ax3.tick_params(axis='y', labelsize=20)
    # Plot 4
    ax4.hist(all_maxes)
    ax4.set_xlabel('Maximum Tanimoto Distance', fontsize=25)
    ax4.set_ylabel('Frequency', fontsize=25)
    fig.savefig(result_dir_plots, facecolor='w')
    # Plot 4
    ax5.hist(all_mins)
    ax5.set_xlabel('Minimum Tanimoto Distance', fontsize=25)
    ax5.set_ylabel('Frequency', fontsize=25)
    fig.savefig(result_dir_plots, facecolor='w')


if __name__ == '__main__':
    calculate_tanimoto("target1")
    calculate_tanimoto("target2")
    calculate_tanimoto("target3")
    calculate_tanimoto("nsp")
    calculate_tanimoto("nsp_sam")
    calculate_tanimoto("spike")
    calculate_tanimoto("ace")
    calculate_tanimoto("spike")
