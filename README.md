# Leveraging Expert Consistency
This repository implements the methodology proposed in [Leveraging Expert Consistency to Improve Algorithmic Decision Support](https://arxiv.org/pdf/2101.09648.pdf). The proposed approach leverages influence to estimate the consistency of multiple experts. This information is then leveraged to learn from experts when there is inferred expert consistency, and from observed labels otherwise.

## How to use ?
```python
# Create model for f_D to model human decision
f_D = BinaryMLP(**params)
f_D = f_D.fit(X_train, D_train, H_train)

# Estimate influence on training set 
folds, predictions, influence = influence_cv(BinaryMLP, X_train, D_train, H_train, params = params, l1_penalties = [0.001, 0.01, 0.1, 1])

# Compute metrics to estimate consistency among experts
center_metric, opposing_metric = compute_agreeability(influence, predictions)

# Amalgamate observed decision and outcomes
high_conf = (predictions > (1 - delta)) | (predictions < delta)

flat_influence = (np.abs(influence) > gamma_3).sum(0) == 0
high_agr = (((center_metric > gamma_1) & (opposing_metric > gamma_2)) | flat_influence) & high_conf
high_agr_correct = ((predictions - tar_train['D']).abs() < delta) & high_agr

A = Y.copy()
A[high_agr_correct] = D_train[high_agr_correct]
index_amalg = (D == 1) | high_agr_correct # Selective labels

# Train a model for the amalgameted outcomes
f_A = BinaryMLP(**params)
f_A = f_A.fit(X_train[index_amalg], A[index_amalg], H[index_amalg])
```

A full example on the MIMIC dataset is described in `example/Triage - MIMIC.ipynb` and a tutorial described how to choose the hyperparameter in `exampl/Tutorial.ipynb`

## Reproduce paper's MIMIC results
To reproduce the paper's results:

0. Clone the repository with dependencies: `git clone git@github.com:XX/influence-labelers.git`.
1. Create a conda environment with all necessary libraries `pytorch`, `pandas`, `numpy`.
2. Download the MIMIC ED dataset and extracts data following `data/Triage - MIMIC - Preprocessing.ipynb`.
3. Then run the comparison following `example/Triage - MIMIC.ipynb` -- `example/k_experiments.py` allows to run k iterations of this same set of experiments in command line.
5. Analyse the results using `4. Analysis - MIMIC.ipynb`.

# Setup
## Structure

### model
This folder contains the neural network model, influence function and amalgamation functions necessary to implement the model.

### example
This folder contains the script used to run the experiment and analyze the results.

### data
This folder contains the script to extract the vital signs from MIMIC ED dataset and generate the semi synthetic labels.

## Clone
```
git clone git@github.com:XX/influence-labelers.git
```

## Requirements
The model relies on  `pytorch`, `pandas`, `numpy` and `sklearn`.  
To analyze the results `matplotlib` is necessary.
