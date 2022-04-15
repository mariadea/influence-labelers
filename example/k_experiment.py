# This script allows to run k experiments on a given dataset

# Parse command and additional parameters
import argparse
parser = argparse.ArgumentParser(description = 'Running k experiments of amalgamation.')
parser.add_argument('--dataset', '-d', type = str, default = 'mimic', help = 'Dataset to analyze (child, mimic or mimic_synth).', choices = ['child', 'mimic', 'mimic_synth'])
parser.add_argument('-k', type = int, default = 1, help = 'Number of iterations to run.')
parser.add_argument('--selective', '-s', action='store_true', help = 'Run under selective labels.')

args = parser.parse_args()

print('Script running on {} for {} iterations'.format(args.dataset , args.k))

params = {'layers': []} # If = [] equivalent to a simple logistic regression
l1_penalties = [0.001, 0.01, 0.1, 1, 5, 10.]

rho = 0.05 # Control which point to consider from a confience point of view
tau = 1.0  # Balance between observed and expert labels

# Open dataset and update
import pandas as pd
import pickle as pkl
import numpy as np

if args.dataset == 'mimic':
    pi_1 = 2.8 # Control criterion on centre mass metric
    pi_2 = 0.95 # Control criterion on opposing metric
    data_set = "../data/triage_semi_synthetic.csv" 
    triage = pd.read_csv(data_set, index_col = [0, 1])
    covariates, target, experts = triage.drop(columns = ['D', 'Y1', 'Y2', 'YC', 'acuity', 'nurse']), triage[['D', 'Y1', 'Y2', 'YC']], triage['nurse']

elif args.dataset == 'mimic_synth':
    pi_1 = 2.8 # Control criterion on centre mass metric
    pi_2 = 0.95 # Control criterion on opposing metric
    data_set = "../data/triage_clean.csv" 
    triage = pd.read_csv(data_set, index_col = [0, 1])
    covariates, target, experts = triage.drop(columns = ['D', 'Y1', 'Y2', 'YC', 'acuity', 'nurse']), triage[['D', 'Y1', 'Y2', 'YC']], triage['nurse']

elif args.dataset == 'child':
    pi_1 = 4 # Control criterion on centre mass metric
    pi_2 = 0.8 # Control criterion on opposing metric
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
    refer_ids = np.delete(refer_ids, drop_idx, axis = 0).flatten()
    screener_ids = np.delete(screener_ids, drop_idx, axis = 0)

    D = D.reshape((D.shape[0],))
    Y_obs = Y_obs.reshape((Y_obs.shape[0],))

    target = pd.DataFrame({'D': D, 'Y1': Y_obs, 'Y2': Y_serv, 'Y3': Y_sub}, index = refer_ids)
    experts = pd.Series(screener_ids, index = refer_ids)
    covariates = pd.DataFrame(X, index = refer_ids)


# Iterate k times the algorithm
import sys
sys.path.append('../')

from model import *
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

results = []
# Monte Carlo cross validation
for k in range(args.k):
    print("Running iteration {} / {}".format(k, args.k))

    # Split data
    cov_train, cov_test, tar_train, tar_test, nur_train, nur_test = train_test_split(covariates, target, experts, test_size = 0.2, random_state = k)

    # Train on decision
    for l1_penalty in l1_penalties:
        try:
            model = BinaryMLP(**params)
            model = model.fit(cov_train, tar_train['D'], nur_train, l1_penalty = l1_penalty, check = True)
            break
        except Exception as e:
            print(e, l1_penalties.pop(0))
            pass

    Y_pred_h_test = pd.Series(model.predict(cov_test).flatten(), index = cov_test.index, name = 'Human')


    # Fold evaluation of influences
    folds, predictions, influence = influence_cv(BinaryMLP, cov_train, tar_train['D'], nur_train, params = params, l1_penalties = l1_penalties)
    center_metric, opposing_metric = compute_agreeability(influence)
    
    # Amalgamation
    high_conf = (predictions > (1 - rho)) | (predictions < rho)
    high_agr = (center_metric > pi_1) & (opposing_metric > pi_2) & high_conf
    high_agr_correct = ((predictions - tar_train['D']).abs() < rho) & high_agr

    tar_train['Ya'] = tar_train['Y1'].astype(int)
    tar_train['Ya'][high_agr_correct] = (1 - tau) * tar_train['Y1'][high_agr_correct] \
                                        + tau * tar_train['D'][high_agr_correct]

    index_amalg = (tar_train['D'] | high_agr_correct) if args.selective else tar_train['D'].isin([0, 1])


    # Amalgamation model
    model = BinaryMLP(**params)
    model = model.fit(cov_train[index_amalg], tar_train[index_amalg]['Ya'], nur_train[index_amalg], l1_penalty = l1_penalty)
    Y_pred_amalg_test = pd.Series(model.predict(cov_test).flatten(), index = cov_test.index, name = 'Amalgamation')


    # Observed outcome
    index_observed = tar_train['D'] if args.selective else tar_train['D'].isin([0, 1])
    model = BinaryMLP(**params)
    model = model.fit(cov_train[index_observed], tar_train['Y1'][index_observed], nur_train[index_observed], l1_penalty = l1_penalty)
    Y_pred_obs_test = pd.Series(model.predict(cov_test).flatten(), index = cov_test.index, name = 'Observed')

    results.append(pd.concat([Y_pred_obs_test, Y_pred_amalg_test, Y_pred_h_test], axis = 1))

pkl.dump(results, open('../results/{}_{}.pkl'.format(args.dataset, args.k), 'wb'))