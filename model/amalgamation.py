from multiprocessing.sharedctypes import Value
import torch
import torch.nn as nn
from torch.linalg import solve
from torch.autograd import grad

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .loss import compute_loss

def compute_influence(model, grad_p, x_h, y_h, hessian_train, l1_penalty = 0.001, cutting_threshold = 0):
    theta = model.get_last_weights()

    # Compute impact on training of one user
    grad_h, = grad(compute_loss(model, x_h, y_h, l1_penalty = l1_penalty), theta, create_graph = True)
    grad_h = grad_h[theta.abs() > cutting_threshold].squeeze()

    # Inverse hessian and multiply
    hess_grad = solve(hessian_train, grad_h) #TODO: Approximate instead as in https://github.com/nimarb/pytorch_influence_functions

    return torch.matmul(grad_p, hess_grad)

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

    # Shuffle data - Need separation from fold to ensure group
    sort = np.arange(len(h))
    np.random.seed(42)
    np.random.shuffle(sort)
    resort = np.zeros_like(sort)
    resort[sort] = np.arange(len(h))
    x, y, h = x[sort], y[sort], h[sort]
    
    # Create groups of observations to ensure one expert in each fold
    g, unique_h = np.zeros_like(h), np.unique(h)
    for expert in unique_h:
        selection = h == expert
        g[selection] = np.arange(np.sum(selection))

    splitter = StratifiedGroupKFold(n_split, shuffle = False)
    folds, predictions, influence = np.zeros(len(x)), np.zeros(len(x)), np.zeros((len(unique_h), x.shape[0]))
    for i, (train_index, test_index) in enumerate(splitter.split(x, y, g)):
        folds[test_index] = i
        predictions[test_index], influence[:len(np.unique(h[train_index])), test_index] = influence_estimate(model, 
            x[train_index], y[train_index], h[train_index], x[test_index], l1_penalties = l1_penalties, params = params,
            groups = None if groups is None else groups[train_index])

    return folds[resort], predictions[resort], influence[:, resort]

def center_mass(influence_point):
    inf_sorted = np.sort(np.abs(influence_point))[::-1]
    total = np.sum(inf_sorted)
    center = np.dot(inf_sorted, np.arange(len(influence_point))) / total if total > 0 else 1.
    return center

def opposing(influence_point):
    inf_pos = influence_point[np.where(influence_point > 0)]
    inf_neg = influence_point[np.where(influence_point < 0)]

    total = inf_pos.sum() - inf_neg.sum() # Sum of absolute values
    return np.max([inf_pos.sum(), - inf_neg.sum()]) / total if total > 0 else 1.

def compute_agreeability(influence):
    """
        Compute agreeability of the influence matrix

        Args:
            influence (Matrix np.array): Testing influence of each expert on each point

        Returns:
            (cm, op): Arrays of center of mass and opposing metrics
    """
    cm_inf = np.apply_along_axis(center_mass, 0, influence) # Compute over the different points
    op_inf = np.apply_along_axis(opposing, 0, influence)
    return cm_inf, op_inf
