# Swift Dock


In this study, various machine learning (ML) models were tested in an effort to forecast the docking scores of a ligand for a specific target protein, bypassing the need for detailed docking calculations. The primary objective is to identify a regression model capable of accurately determining the docking scores of ligands within a chemical library, relative to a target protein. This would be achieved using data derived from explicit docking of only a small selection of molecules.

Among the ML models utilized is a neural network model based on Long Short-Term Memory (LSTM). LSTMs are typically deployed in the handling of sequence data within Natural Language Processing (NLP) contexts, such as speech recognition. The specific LSTM model employed in this study is combined with an attention mechanism to enable the neural network model to more effectively distill useful characteristics from incoming ligand data. Pytorch, a widely-used Python ML framework, was utilized to implement the LSTM.

Additionally, several other models were also explored, including XGBoost, which was executed via the XGBoost Python library, as well as decision tree regression and stochastic gradient descent models drawn from the scikit-learn Python library.