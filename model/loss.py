import torch
import torch.nn as nn
from torch.linalg import solve
from torch.autograd import grad
from torch.autograd.functional import jacobian

def compute_loss(model, x, y):
    pred = model(x).view(-1)
    return nn.BCELoss()(pred, y)

def compute_influence(model, grad_p, x_h, y_h, hessian_train):
    theta = model.get_last_weights()

    # Compute impact on training of one user
    grad_h, = grad(compute_loss(model, x_h, y_h), theta, create_graph = True)
    grad_h = grad_h[theta > 0].squeeze()

    # Inverse hessian and multiply
    hess_grad = solve(hessian_train, grad_h) #TODO: Do the inversion before and just multiplication here ?

    return torch.matmul(grad_p, hess_grad)