import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from .loss import compute_loss

def create_mlp(inputdim, layers, activation, dropout):
    """
       Creates a simple multi layer perceptron with a final sigmoid layer

        Args:
            inputdim (int): Input dimension
            layers (int list): List of hidden layers dimension
            activation (str): Activation function (ReLU6, ReLU or Tanh)
            dropout (float): Percentage of dropout used

        Returns:
            nn.Sequential: Torch Neural Network
    """

    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'Tanh':
        act = nn.Tanh()
    else:
        raise ValueError('Unknown activation function')

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias = True))
        if dropout > 0:
            modules.append(nn.Dropout(p = dropout))
        modules.append(act)
        prevdim = hidden

    modules[-1] = nn.Sigmoid()

    return nn.Sequential(*modules)

def train_mlp(model,
            x_train, y_train, x_valid, y_valid, l1_penalty = 0.,
            n_iter = 1000, lr = 1e-3, weight_decay = 0., bs = 100):

    # Separate oprimizer as one might need more time to converge
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    patience, best_loss, previous_loss = 0, np.inf, np.inf
    best_param = deepcopy(model.state_dict())
    
    nbatches = int(x_train.shape[0] / bs) + 1
    index = np.arange(x_train.shape[0])
    t_bar = tqdm(range(n_iter))

    for _ in t_bar:
        np.random.shuffle(index)
        model.train()
        for j in range(nbatches):
            xb = x_train[index[j*bs:(j+1)*bs]]
            yb = y_train[index[j*bs:(j+1)*bs]]
            
            if xb.shape[0] == 0:
                continue

            optimizer.zero_grad()
            loss = compute_loss(model, xb, yb, l1_penalty) 
            loss.backward()
            optimizer.step()

        model.eval()
        xb, yb = x_valid, y_valid
        valid_loss = compute_loss(model, xb, yb, l1_penalty).item() 
        t_bar.set_description("Loss: {:.3f}".format(valid_loss))

        if valid_loss < previous_loss:
            patience = 0
            if valid_loss < best_loss:
                # Update model
                best_loss = valid_loss
                best_param = deepcopy(model.state_dict())
        elif patience == 3:
            break
        else:
            patience += 1

        previous_loss = valid_loss

    # Reload the best weight scheme
    model.load_state_dict(best_param)
    return model