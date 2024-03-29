# This script allows to run k experiments on a given dataset

# Parse command and additional parameters
import argparse
parser = argparse.ArgumentParser(description = 'Running k experiments of amalgamation.')
parser.add_argument('--dataset', '-d', type = str, default = 'mimic', help = 'Dataset to analyze.')
parser.add_argument('-k', type = int, default = 10, help = 'Number of iterations to run.')
parser.add_argument('-s', action='store_true', help = 'Selective labels')
parser.add_argument('--log', '-l', action='store_true', help = 'Run a logistic regression model (otherwise neural network).')
parser.add_argument('-delta', default = 0.05, type = float, help = 'Control which point to consider from a confience point of view.')
parser.add_argument('-gamma1', default = 6, type = float, help = 'Threshold on center mass.')
parser.add_argument('-gamma2', default = 0.95, type = float, help = 'Threshold on opposing.')
parser.add_argument('-gamma3', default = 0.002, type = float, help = 'Threshold on flat influence. Default: ignore.')
args = parser.parse_args()

print('Script running on {} for {} iterations'.format(args.dataset , args.k))

params = {'layers': [[]] if args.log else [[50] * layer for layer in [1, 2, 3]]}  # If = [] equivalent to a simple logistic regression
l1_penalties = [0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000.]
tau = 1.0  # Balance between observed and expert labels

# Open dataset and update
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.model_selection import ShuffleSplit

data_set = "../data/triage_scenario_{}.csv".format(args.dataset[args.dataset.index('_') + 1:]) 
triage = pd.read_csv(data_set, index_col = [0, 1])
splitter, groups = ShuffleSplit(n_splits = args.k, train_size = .75, random_state = 42), None
covariates, target, experts = triage.drop(columns = ['D', 'Y1', 'Y2', 'YC', 'nurse']), triage[['D', 'Y1', 'Y2', 'YC']], triage['nurse']

selective = args.s

# Create results folder
import os
path_results = '../results/{}_{}_delta={}_gamma1={}_gamma2={}_gamma3={}{}/'.format(args.dataset, 'log' if args.log else 'mlp', args.delta, args.gamma1, args.gamma2, args.gamma3, '_selective' if selective else '')
os.makedirs(path_results, exist_ok = True)

# Iterate k times the algorithm
import sys
sys.path.append('../')

from model import *
from model.defer import DeferMLP

results = []
amalgamation = []
# Monte Carlo cross validation
for k, (train, test) in enumerate(splitter.split(covariates, target, groups)):
    # Create folder for iteration
    path_fold = path_results + 'fold_{}/'.format(k)
    try:
        os.makedirs(path_fold)
    except:
        # Skip fold if folder already exists
        continue

    print("Running iteration {} / {}".format(k + 1, args.k))

    # Split data
    cov_train, cov_test, tar_train, tar_test, nur_train, nur_test = covariates.iloc[train], \
            covariates.iloc[test], target.iloc[train], target.iloc[test], \
            experts.iloc[train], experts.iloc[test]

    # Train on decision
    f_D = BinaryMLP(**params)
    f_D = f_D.fit(cov_train, tar_train['D'], nur_train, platt_calibration = True, groups = None if groups is None else groups[train])
    pred_D_test = pd.Series(f_D.predict(cov_test), index = cov_test.index)
    pred_D_test.to_csv(path_fold + 'f_D.csv')

    # Proposed Approach
    ## Fold evaluation of influences
    try:
        folds, predictions, influence = influence_cv(BinaryMLP, cov_train, tar_train['D'], nur_train, params = params, l1_penalties = l1_penalties, groups = None if groups is None else groups[train])
    except:
        print('Iteration {} - Not invertible hessian'.format(k))
        continue
    center_metric, opposing_metric = compute_agreeability(influence, predictions)
    
    ## Amalgamation
    flat_influence = (np.abs(influence) > args.gamma3).sum(0) == 0
    high_conf = (predictions > (1 - args.delta)) if args.dataset == 'child' else ((predictions > (1 - args.delta)) | (predictions < args.delta))
    high_agr = (((center_metric > args.gamma1) & (opposing_metric > args.gamma2)) | flat_influence) & high_conf
    high_agr_correct = (((predictions - tar_train['D']).abs() < args.delta) & high_agr)

    ya = tar_train['Y1'].copy().astype(int)
    ya.loc[high_agr_correct] = (1 - tau) * tar_train.loc[high_agr_correct, 'Y1'].copy() \
                             + tau * tar_train.loc[high_agr_correct, 'D'].copy()

    index_amalg = ((tar_train['D'] == 1) | high_agr_correct) if selective else tar_train['D'].isin([0, 1])

    ## Train model on new labels
    f_A = BinaryMLP(**params)
    f_A = f_A.fit(cov_train[index_amalg], ya[index_amalg], nur_train[index_amalg], groups = None if groups is None else groups[train][index_amalg])
    pd.Series(f_A.predict(cov_test), index = cov_test.index).to_csv(path_fold + 'f_A.csv')
    indicator = pd.Series(False, index = cov_train.index)
    indicator.loc[high_agr_correct] = True
    pd.concat([ya.rename('Label'), indicator.rename('Indicator')], axis = 1).to_csv(path_fold + 'amalgamation.csv')

    # Observed outcome
    index_observed = tar_train['D'] == 1 if selective else tar_train['D'].isin([0, 1])
    f_Y = BinaryMLP(**params)
    f_Y = f_Y.fit(cov_train[index_observed], tar_train['Y1'][index_observed], nur_train[index_observed], groups = None if groups is None else groups[train][index_observed])
    pred_Y_test = pd.Series(f_Y.predict(cov_test), index = cov_test.index)
    pred_Y_test.to_csv(path_fold + 'f_Y.csv')



    # Baselines
    # Hybrid model: initialize rely on humans
    pred_hyb_test = pred_D_test.copy()

    ## Compute which test points are part of A for test set
    predictions_test, influence_test = influence_estimate(BinaryMLP, cov_train, tar_train['D'], nur_train, cov_test, params = params, l1_penalties = l1_penalties, groups = None if groups is None else groups[train])
    center_metric, opposing_metric = compute_agreeability(influence_test, predictions_test)
    flat_influence_test = (np.abs(influence_test) > args.gamma3).sum(0) == 0
    high_conf_test = (predictions_test > (1 - args.delta)) if args.dataset == 'child' else ((predictions_test > (1 - args.delta)) | (predictions_test < args.delta))
    high_agr_test = (((center_metric > args.gamma1) & (opposing_metric > args.gamma2)) | flat_influence_test) & high_conf_test
    high_agr_correct_test = ((predictions_test - tar_test['D']).abs() < args.delta) & high_agr_test

    ## Retrain a model on non almagamation only and calibrate: Rely on observed
    f_hyb = BinaryMLP(**params)
    f_hyb = f_hyb.fit(cov_train[index_observed], tar_train['Y1'][index_observed], nur_train[index_observed], platt_calibration = True, groups = None if groups is None else groups[train][index_observed])
    pred_hyb_test.loc[~high_agr_correct_test] = f_hyb.predict(cov_test.loc[~high_agr_correct_test])
    pred_hyb_test.to_csv(path_fold + 'f_hyb.csv')

    # Deferal model
    f_def = DeferMLP(**params)
    f_def = f_def.fit(cov_train[index_observed], tar_train['Y1'][index_observed], tar_train['D'][index_observed], groups = None if groups is None else groups[train][index_observed])
    pd.Series(f_def.predict(cov_test, tar_test['D']), index = cov_test.index).to_csv(path_fold + 'f_def.csv')

    # Average predictions
    ((pred_Y_test + pred_D_test) / 2).to_csv(path_fold + 'f_ensemble.csv')

    # Weak supervision
    weak_labels = (tar_train['D'] + tar_train['Y1']).fillna(tar_train['D']) / 2
    f_weak = BinaryMLP(**params)
    f_weak = f_weak.fit(cov_train, weak_labels, nur_train)
    pd.Series(f_weak.predict(cov_test), index = cov_test.index).to_csv(path_fold + 'f_weak.csv')

    # Noisy labels learning
    ## Clean labels
    import cleanlab
    from sklearn.neural_network import MLPClassifier
    f_robust = cleanlab.classification.CleanLearning(MLPClassifier(50))
    label_issues = f_robust.find_label_issues(cov_train, tar_train['D'].astype(int))
    selection = ~label_issues.is_label_issue.values

    ## Train on subset
    f_robust = BinaryMLP(**params)
    f_robust.fit(cov_train.iloc[selection], tar_train['D'].iloc[selection], nur_train)
    pd.Series(f_robust.predict(cov_test), index = cov_test.index).to_csv(path_fold + 'f_robust.csv')
