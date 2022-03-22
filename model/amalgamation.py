import torch
import torch.nn as nn
from torch.linalg import solve
from torch.autograd import grad

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .loss import compute_loss

def compute_influence(model, grad_p, x_h, y_h, hessian_train):
    theta = model.get_last_weights()

    # Compute impact on training of one user
    grad_h, = grad(compute_loss(model, x_h, y_h), theta, create_graph = True)
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
    splitter = StratifiedKFold(split, shuffle = True, random_state = 42)
    folds, predictions, influence = np.zeros(len(x)), np.zeros(len(x)), np.zeros((len(np.unique(h)), x.shape[0]))
    for i, (train_index, test_index) in enumerate(splitter.split(x, y)):
        folds[train_index] = i

        # Train model on the subset
        model_cv = model(**params)
        model_cv.fit(x[train_index], y[train_index], h[train_index], **fit_params)

        # TODO: Calibrate NN
        predictions[test_index] = model_cv.predict(x[test_index])[:, 0]

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
