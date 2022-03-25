import torch
import torch.nn as nn

def compute_loss(model, x, y, lbd = 0.001):
    pred = model(x).view(-1)

    # L2 regularization to avoid colineartity of the hessian
    regularizer, params = 0, 0
    for param in model.parameters():
        regularizer += torch.norm(param, p = 1)
        params += 1
    regularizer /= params

    return nn.BCELoss()(pred, y) + lbd * regularizer