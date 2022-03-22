import torch.nn as nn

def compute_loss(model, x, y):
    pred = model(x).view(-1)
    return nn.BCELoss()(pred, y)