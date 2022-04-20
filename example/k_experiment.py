# This script allows to run k experiments on a given dataset

# Parse command and additional parameters
import argparse
parser = argparse.ArgumentParser(description = 'Running k experiments of amalgamation.')
parser.add_argument('--dataset', '-d', type = str, default = 'mimic', help = 'Dataset to analyze (child, mimic or mimic_synth).', choices = ['child', 'mimic', 'mimic_1', 'mimic_2', 'mimic_3', 'mimic_4'])
parser.add_argument('-k', type = int, default = 20, help = 'Number of iterations to run.')
parser.add_argument('--log', '-l', action='store_true', help = 'Run a logistic regression model (otherwise neural network).')
parser.add_argument('-p1', default = 2.8, type = float, help = 'Threshold on center mass.')
parser.add_argument('-p2', default = 0.95, type = float, help = 'Threshold on opposing.')
parser.add_argument('-p3', default = 0., type = float, help = 'Threshold on flat influence. Default: ignore.')
args = parser.parse_args()

print('Script running on {} for {} iterations'.format(args.dataset , args.k))

params = {'layers': [[]] if args.log else [[node] * layer for node in [10, 100] for layer in [1, 2, 3]]}  # If = [] equivalent to a simple logistic regression
l1_penalties = [0.001, 0.01, 0.1, 1., 10., 100., 1000., 10000.]

rho = 0.05 # Control which point to consider from a confience point of view
tau = 1.0  # Balance between observed and expert labels

# Open dataset and update
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

if args.dataset == 'mimic':
    data_set = "../data/triage_clean.csv" 
    triage = pd.read_csv(data_set, index_col = [0, 1])
    splitter, groups = ShuffleSplit(n_splits = args.k, train_size = .75, random_state = 42), None
    covariates, target, experts = triage.drop(columns = ['D', 'Y1', 'Y2', 'YC', 'acuity', 'nurse']), triage[['D', 'Y1', 'Y2', 'YC']], triage['nurse']

    selective = False

elif '_' in args.dataset:
    data_set = "../data/triage_scenario_{}.csv".format(args.dataset[args.dataset.index('_') + 1:]) 
    triage = pd.read_csv(data_set, index_col = [0, 1])
    splitter, groups = ShuffleSplit(n_splits = args.k, train_size = .75, random_state = 42), None
    covariates, target, experts = triage.drop(columns = ['D', 'Y1', 'Y2', 'YC', 'acuity', 'nurse']), triage[['D', 'Y1', 'Y2', 'YC']], triage['nurse']

    selective = False

elif args.dataset == 'child':
    with open('../../data/ChildWelfare/X_preprocess.pkl', 'rb') as handle:
        X, screener_ids, refer_ids, Y_obs, D, Y_serv,Y_sub,colnames = pkl.load(handle)

    # Remove less than 10 observation by experts
    drop_experts = []
    for num in screener_ids:
        if screener_ids.count(num) < 10:
            drop_experts.append(num)

    drop_idx = []
    for index, elem in enumerate(screener_ids):
        if elem in drop_experts:
            drop_idx.append(index)
    
    X = np.delete(X, drop_idx, axis = 0)
    Y_serv = np.delete(Y_serv, drop_idx, axis = 0)
    Y_sub = np.delete(Y_sub, drop_idx, axis = 0)
    Y_obs = np.delete(Y_obs, drop_idx, axis = 0)
    D = np.delete(D, drop_idx, axis = 0)
    groups = np.delete(refer_ids, drop_idx, axis = 0).flatten()
    screener_ids = np.delete(screener_ids, drop_idx, axis = 0)

    D = D.reshape((D.shape[0],))
    Y_obs = Y_obs.reshape((Y_obs.shape[0],))

    target = pd.DataFrame({'D': D, 'Y1': Y_obs, 'Y2': Y_serv, 'Y3': Y_sub})
    experts = pd.Series(screener_ids)
    covariates = pd.DataFrame(X[:, :-1]) # Remove 0 columns
    splitter = GroupShuffleSplit(n_splits = args.k, train_size = .75, random_state = 42)

    selective = True

# Iterate k times the algorithm
import sys
sys.path.append('../')

from model import *

results = []
# Monte Carlo cross validation
for k, (train, test) in enumerate(splitter.split(covariates, target, groups)):
    print("Running iteration {} / {}".format(k + 1, args.k))

    # Split data
    cov_train, cov_test, tar_train, tar_test, nur_train, nur_test = covariates.iloc[train], \
            covariates.iloc[test], target.iloc[train], target.iloc[test], \
            experts.iloc[train], experts.iloc[test]

    # Train on decision
    model = BinaryMLP(**params)
    model = model.fit(cov_train, tar_train['D'], nur_train, platt_calibration = True, groups = None if groups is None else groups[train])
    pred_h_test = pd.Series(model.predict(cov_test), index = cov_test.index, name = 'Human')


    # Fold evaluation of influences
    try:
        folds, predictions, influence = influence_cv(BinaryMLP, cov_train, tar_train['D'], nur_train, params = params, l1_penalties = l1_penalties, groups = None if groups is None else groups[train])
    except:
        print('Iteration {} - Not invertible hessian'.format(k))
        continue
    center_metric, opposing_metric = compute_agreeability(influence)
    
    # Amalgamation
    flat_influence = (np.abs(influence) > args.p3).sum(0) == 0
    high_conf = (predictions > (1 - rho)) | (predictions < rho)
    high_agr = (center_metric > args.p1) & (opposing_metric > args.p2) & high_conf
    high_agr_correct = (((predictions - tar_train['D']).abs() < rho) & high_agr) | flat_influence

    tar_train.loc[:, 'Ya'] = tar_train['Y1'].copy().astype(int)
    tar_train.loc[high_agr_correct, 'Ya'] = (1 - tau) * tar_train['Y1'][high_agr_correct].copy() \
                                                + tau * tar_train['D'][high_agr_correct].copy()

    index_amalg = ((tar_train['D'] == 1) | high_agr_correct) if selective else tar_train['D'].isin([0, 1])


    # Amalgamation model
    model = BinaryMLP(**params)
    model = model.fit(cov_train[index_amalg], tar_train[index_amalg]['Ya'], nur_train[index_amalg], groups = None if groups is None else groups[train][index_amalg])
    pred_amalg_test = pd.Series(model.predict(cov_test), index = cov_test.index, name = 'Amalgamation')


    # Observed outcome
    index_observed = tar_train['D'] == 1 if selective else tar_train['D'].isin([0, 1])
    model = BinaryMLP(**params)
    model = model.fit(cov_train[index_observed], tar_train['Y1'][index_observed], nur_train[index_observed], groups = None if groups is None else groups[train][index_observed])
    pred_obs_test = pd.Series(model.predict(cov_test), index = cov_test.index, name = 'Observed')


    # Hybrid model: initialize rely on humans
    pred_hyb_test = pred_h_test.copy().rename('Hybrid')

    # Compute which test points are part of A for test set
    predictions_test, influence_test = influence_estimate(BinaryMLP, cov_train, tar_train['D'], nur_train, cov_test, params = params, l1_penalties = l1_penalties, groups = None if groups is None else groups[train])
    center_metric, opposing_metric = compute_agreeability(influence_test)
    high_conf_test = (predictions_test > (1 - rho)) if args.dataset == 'child' else ((predictions_test > (1 - rho)) | (predictions_test < rho))
    high_agr_test = (center_metric > args.p1) & (opposing_metric > args.p2) & high_conf_test
    high_agr_correct_test = ((predictions_test - tar_test['D']).abs() < rho) & high_agr_test

    # Retrain a model on non almagamation only and calibrate: Rely on observed
    model = BinaryMLP(**params)
    model = model.fit(cov_train[index_observed], tar_train['Y1'][index_observed], nur_train[index_observed], platt_calibration = True, groups = None if groups is None else groups[train][index_observed])
    pred_hyb_test.loc[~high_agr_correct_test] = model.predict(cov_test.loc[~high_agr_correct_test])

    results.append(pd.concat([pred_obs_test, pred_amalg_test, pred_h_test, pred_hyb_test], axis = 1))

    pkl.dump(results, open('../results/{}_{}_p1={}_p2=_{}_p3={}.pkl'.format(args.dataset, 'log' if args.log else 'mlp', args.p1, args.p2, args.p3), 'wb'))
