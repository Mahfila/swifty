# Swift Dock


In this study, various machine learning (ML) models were tested in an effort to forecast the docking scores of a ligand for a specific target protein, bypassing the need for detailed docking calculations. The primary objective is to identify a regression model capable of accurately determining the docking scores of ligands within a chemical library, relative to a target protein. This would be achieved using data derived from explicit docking of only a small selection of molecules.

Among the ML models utilized is a neural network model based on Long Short-Term Memory (LSTM). LSTMs are typically deployed in the handling of sequence data within Natural Language Processing (NLP) contexts, such as speech recognition. The specific LSTM model employed in this study is combined with an attention mechanism to enable the neural network model to more effectively distill useful characteristics from incoming ligand data. Pytorch, a widely-used Python ML framework, was utilized to implement the LSTM.

Additionally, several other models were also explored, including XGBoost, which was executed via the XGBoost Python library, as well as decision tree regression and stochastic gradient descent models drawn from the scikit-learn Python library.


# Setting up the environment

1. Make sure Python 3.7 is installed on your system
2. Create a virtual environment and run  pip install -r requirements.txt
3. Under swifty, run sudo chmod -R 777 logs
## Training Using LSTM
### First build a model and get validation results 
 There are many options to train the lstm model depending on the target, descriptor and training_size of your choice, you can also use 5 fold cross validation. So lets you have a csv file named 'docking_scores.csv' and you want use the mac descriptor and using 50 molecules selected randomly with 5 fold cross validation. Use this steps.
1. Put your target in the dataset folder. You should use the format of the sample test.csv target in the dataset folder.
2. Example -  src/models run `python main_lstm.py --input docking_scores.csv --descriptors mac --training_sizes 50 --cross_validate True`. This will train a model using 50 molecules selected randomly from target 'docking_scores.csv' file using mac descriptor without 5 cross validation
3. Above code will produce result directory with 5 folders. Each of the file names in these folders has a name beginning with this format - lstm_target_descriptor_training_size.
- project_info - contains information such as training size and training times - Has a format of {lstm}_{input_file_name}_{descriptor}_{training_size}_project_info.csv
- serialized_models - contains the trained model after training -  Has a format of {lstm}_{input_file_name}_{descriptor}_{training_size}_model.pt
- test_predictions - contains each docking score and its model prediction - Has a format of {lstm}_{input_file_name}_{descriptor}_{training_size}_test_predictions.csv
- testing_metrics - contains metrics (R-squared, mean absolute error) in testing - Has a format of {lstm}_{input_file_name}_{descriptor}_{training_size}_test_metrics.csv
- validation_metrics - contains metrics (R-squared, mean absolute error) in 5 fold cross validation - This result is only created when you set -- cross_validation True  - Has a format of {lstm}_{input_file_name}_{descriptor}_{training_size}_validation_metrics.csv

Note that docking_scores.csv file should contain the following columns separated by comma.
- first column: docking score
- second column: smile


### Making Prediction for your target using LSTM 
Example, Under src/models directory, run the following command:  
`python lstm_inference.py --input_file ../../datasets/test.csv  --output_dir ../../results/prediction_results --model_name ../../results/serialized_models/lstm_target_mac_50_model.pt`
  -  --input_file - This is the file that contains molecules you want to predict the docking scores of. Make sure your input csv has the same format at molecules_for_prediction.csv (containing smiles of molecules) in the dataset folder. This file should be in the datasets folder
  -  --output_dir - Where you want the results to be saved
  -  --model_name - The model_name is the path to the model of your choice that you get after training that is saved in the previous step. Example 'lstm_target_mac_50_model.pt'. Make sure this file is in the serialized_models_models directory


## Training Using other models (from scikit-learn)
To train the models using other models other than lstm, first create the dataset. For docking_scores.csv file that contains smiles and docking scores. If for example you want use the mac descriptor and a training size of 50 molecules with 5 fold cross validation. Use these steps.
1. Under src/utils, run `python create_fingerprint_data.py --input docking_scores.csv --descriptors mac`. This will create the dataset for the 'docking_scores.csv' using the mac descriptor.
2. Next, under src/models run `python main_ml.py --input docking_scores.csv --descriptors mac --training_sizes 50 --regressor sgreg`. This will train the sgreg model using the molecules in docking_scores.csv for training size of 50 with for mac descriptor.
3. Above code will produce a result directory with 5 folders. Below are the folders created
- project_info - contains information such as training size and training times - Has a format of {model_name}_{input_file_name}_{descriptor}_{training_size}_project_info.csv
- serialized_models - contains the trained model after training -  Has a format of {model_name}_{input_file_name}_{descriptor}_{training_size}_model.pt
- test_predictions - contains each docking score and its model prediction - Has a format of {model_name}_{input_file_name}_{descriptor}_{training_size}_test_predictions.csv
- testing_metrics - contains metrics (R-squared, mean absolute error) in testing - Has a format of {model_name}_{input_file_name}_{descriptor}_{training_size}_test_metrics.csv
- validation_metrics - contains metrics (R-squared, mean absolute error) in 5 fold cross validation - This result is only created when you set -- cross_validation True  - Has a format of {model_name}_{input_file_name}_{descriptor}_{training_size}_validation_metrics.csv

# Making Prediction for your target with other models (sgreg, xgboost, decision tree)
1. For the molecules you want to predict the docking scores of. Make sure your input csv has the same format at input.csv in the dataset folder
2. Under src/models, run `python other_models_inference.py --input_file --output_dir --model_name`
- --input_file is the path to the your input -  example is molecules_for_prediction.csv
- --output_dir is the path where is the results are saved
- --model_name is the path to the choice to make the docking score predictions. Example 'sgdreg_docking_scores_mac_50_model.pkl'
