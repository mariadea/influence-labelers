import torch
import torch.nn as nn

def compute_loss(model, x, y, l1_penalty = 0.):
    """
        Compute BCE loss of the model output with a l penalty on the weights
    """
    pred = model(x).view(-1)

    # L2 regularization to avoid colineartity of the hessian
    regularizer = model.get_last_weights().abs().mean()

    return nn.BCELoss()(pred, y) + l1_penalty * regularizer