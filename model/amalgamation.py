import torch
import torch.nn as nn
from torch.linalg import solve
from torch.autograd import grad

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from .loss import compute_loss

def compute_influence(model, grad_p, x_h, y_h, hessian_train, l1_penalty = 0.001):
    theta = model.get_last_weights()

    # Compute impact on training of one user
    grad_h, = grad(compute_loss(model, x_h, y_h, l1_penalty = l1_penalty), theta, create_graph = True)
    grad_h = grad_h[theta > 0].squeeze()

    # Inverse hessian and multiply
    hess_grad = solve(hessian_train, grad_h) #TODO: Approximate instead as in https://github.com/nimarb/pytorch_influence_functions

    return torch.matmul(grad_p, hess_grad)

def influence_cv(model, x, y, h, params = {}, fit_params = {}, split = 3):
    """
    Compute a stratified cross validation to estimate the influence of each points

    Args:
        model (Object): Create a model (need to have predict and influence functions)
        x (np.array pd.DataFrame): Covariates
        y (np.array pd.DataFrame): Associated outcome
        h (np.array pd.DataFrame): Associated expert
        params (Dict): Dictionary to initialize the model with
        fit_params (Dict): Dictionary for training
        split (int): Number of fold used for the stratified computation of influence

    Returns:
        folds, predictions, influence: Arrays of each point fold, predictions by the model and influence (dim len(x) * num experts)
    """
    x, y, h = (x.values, y.values, h.values) if isinstance(x, pd.DataFrame) else (x, y, h)

    # Shuffle data - Need separation from fold to ensure group
    sort = np.arange(len(h))
    x, y, h = x[sort], y[sort], h[sort]

    # Create groups of observations to ensure one expert in each fold
    g, unique_h = np.zeros_like(h), len(np.unique(h))
    for expert in range(unique_h):
        selection = h == expert
        g[selection] = np.arange(np.sum(selection))

    splitter = StratifiedGroupKFold(split, shuffle = True, random_state = 42)
    folds, predictions, influence = np.zeros(len(x)), np.zeros(len(x)), np.zeros((unique_h, x.shape[0]))
    for i, (train_index, test_index) in enumerate(splitter.split(x, y, g)):
        folds[test_index] = i
        train_index, val_index = train_test_split(train_index, test_size = 0.15, shuffle = False)

        # Train model on the subset
        model_cv = model(**params)
        model_cv.fit(x[train_index], y[train_index], h[train_index], **fit_params, val = (x[val_index], y[val_index]))

        # Calibrate NN on validation set - Platt
        pred_val = model_cv.predict(x[val_index])
        pred_test = model_cv.predict(x[test_index])
        calibrated = LogisticRegression().fit(pred_val, y[val_index])
        predictions[test_index] = calibrated.predict_proba(pred_test)[:, 1]

        # Compute influence
        influence[:, test_index] = model_cv.influence(x[test_index])

    return folds, predictions, influence

def center_mass(influence_point):
    inf_sorted = np.sort(np.abs(influence_point))[::-1]
    center = np.dot(inf_sorted, np.arange(len(influence_point))) / np.sum(inf_sorted)
    return center

def opposing(influence_point):
    inf_pos = influence_point[np.where(influence_point > 0)]
    inf_neg = influence_point[np.where(influence_point < 0)]

    total = inf_pos.sum() - inf_neg.sum()
    opposing = np.max([inf_pos.sum(), - inf_neg.sum()]) / total

    return opposing

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
