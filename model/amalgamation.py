from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
from torch.linalg import solve
from torch.autograd import grad

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .loss import compute_loss

def compute_influence(model, grad_p, x_h, y_h, hessian_train, l1_penalty = 0.001, cutting_threshold = 0):
    theta = model.get_last_weights()

    # Compute impact on training of one user
    grad_h, = grad(compute_loss(model, x_h, y_h, l1_penalty = l1_penalty), theta, create_graph = True)
    grad_h = grad_h[theta.abs() > cutting_threshold].squeeze()

    # Inverse hessian and multiply
    hess_grad = solve(hessian_train, grad_h) #TODO: Approximate instead as in https://github.com/nimarb/pytorch_influence_functions

    return - torch.matmul(grad_p, hess_grad)

def influence_estimate(model, x, y, h, x_apply, l1_penalties = [0], params = {}, groups = None):
    """
        Estimate the influence of x_apply after training model on x, y, h

        Args:
            model (Object): Create a model (need to have predict and influence functions)
            x (np.array pd.DataFrame): Covariates
            y (np.array pd.DataFrame): Associated outcome
            h (np.array pd.DataFrame): Associated expert
            x_apply (np.array pd.DataFrame): Covariates of set to compute
            l1_penalties (list float): L1 penalty to explore until inversion of the hessian
            params (Dict): Dictionary to initialize the model with
            fit_params (Dict): Dictionary for training

        Returns:
            predictions, influence: Predictions by the model and influence (dim len(x) * num experts)
    """
    x, y, h = (x.values, y.values, h.values) if isinstance(x, pd.DataFrame) else (x, y, h)

    # Train model
    for l1 in l1_penalties:
        try:
            model_l1 = model(**params)
            model_l1.fit(x, y, h, l1_penalty = l1, check = True, groups = groups, platt_calibration = True)
            break
        except Exception as e:
            print('L1 = {} not large enough'.format(l1))
    else:
        raise ValueError('None of the l1 penalties led to an invertible hessian.')

    return model_l1.predict(x_apply), model_l1.influence(x_apply)

def influence_cv(model, x, y, h, l1_penalties = [0], params = {}, groups = None, n_split = 3):
    """
    Compute a stratified cross validation to estimate the influence of each points

    Args:
        model (Object): Create a model (need to have predict and influence functions)
        x (np.array pd.DataFrame): Covariates
        y (np.array pd.DataFrame): Associated outcome
        h (np.array pd.DataFrame): Associated expert
        l1_penalties (list float): L1 penalty to explore until inversion of the hessian
        params (Dict): Dictionary to initialize the model with
        fit_params (Dict): Dictionary for training
        n_split (int): Number of fold used for the stratified computation of influence

    Returns:
        folds, predictions, influence: Arrays of each point fold, predictions by the model and influence (dim len(x) * num experts)
    """
    x, y, h = (x.values, y.values, h.values) if isinstance(x, pd.DataFrame) else (x, y, h)

    # Create groups of observations to ensure one expert in each fold
    # And that groups are not splitted 
    # => Stratified k fold (ensure each experts in each folds in similar amount) while respecting groups
    splitter = StratifiedKFold(n_split, shuffle = True, random_state = 42) if groups is None else StratifiedGroupKFold(n_split, shuffle = True, random_state = 42)
    folds, predictions, influence = np.zeros(len(x)), np.zeros(len(x)), np.zeros((len(np.unique(h)), x.shape[0]))
    for i, (train_index, test_index) in enumerate(splitter.split(x, h, groups)):
        folds[test_index] = i
        predictions[test_index], influence[:len(np.unique(h[train_index])), test_index] = influence_estimate(model, 
            x[train_index], y[train_index], h[train_index], x[test_index], l1_penalties = l1_penalties, params = params,
            groups = None if groups is None else groups[train_index])

    return folds, predictions, influence

def center_mass(influence_point):
    inf_sorted = np.sort(np.abs(influence_point))[::-1]
    total = np.sum(inf_sorted)
    center = np.dot(inf_sorted, np.arange(len(influence_point))) / total if total > 0 else 1.
    return center

def opposing(influence_point, prediction):
    inf_pos = influence_point[np.where(influence_point > 0)]
    inf_neg = influence_point[np.where(influence_point < 0)]

    total = inf_pos.sum() - inf_neg.sum() # Sum of absolute values
    if total == 0: return 1
    return (inf_pos.sum() if prediction > 0.5 else - inf_neg.sum()) / total 

def compute_agreeability(influence, predictions):
    """
        Compute agreeability of the influence matrix

        Args:
            influence (Matrix np.array): Testing influence of each expert on each point

        Returns:
            (cm, op): Arrays of center of mass and opposing metrics
    """
    cm_inf = np.apply_along_axis(center_mass, 0, influence) # Compute over the different points
    op_inf = np.array([opposing(influence[:, i], p) for i, p in enumerate(predictions)])
    return cm_inf, op_inf


def ensemble_agreement_cv(model, x, y, h, params = {}, groups = None, n_split = 3):
    """
    Compute a stratified cross validation to estimate the influence of each points
    A model is fitted on each experts and then the average decision is estimated

    Args:
        model (Object): Create a model (need to have predict and influence functions)
        x (np.array pd.DataFrame): Covariates
        y (np.array pd.DataFrame): Associated outcome
        h (np.array pd.DataFrame): Associated expert
        params (Dict): Dictionary to initialize the model with
        fit_params (Dict): Dictionary for training
        n_split (int): Number of fold used for the stratified computation of influence

    Returns:
        decisions: Decisions predictions for each experts (dim num experts * len(x))
    """
    x, y, h = (x.values, y.values, h.values) if isinstance(x, pd.DataFrame) else (x, y, h)

    splitter = StratifiedKFold(n_split, shuffle = True, random_state = 42) if groups is None else StratifiedGroupKFold(n_split, shuffle = True, random_state = 42)
    decisions = np.zeros((len(np.unique(h)), len(x)))
    for (train_index, test_index) in splitter.split(x, h, groups):
        for i, expert in enumerate(np.unique(h)):
            model_expert = model(**params)
            selection = h[train_index] == expert
            model_expert.fit(x[train_index][selection], y[train_index][selection], h[train_index][selection], platt_calibration = np.abs(y[train_index][selection].mean() - 1) != 0.5)
            decisions[i, test_index] = model_expert.predict(x[test_index])

    return decisions