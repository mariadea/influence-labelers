"""
    Defer model from Madras et al.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
from .utils import create_mlp
from sklearn.model_selection import ParameterGrid, ShuffleSplit, GroupShuffleSplit

class DeferMLP:
    """
        Model API to call for using the method
        Preprocess data to shape it to the right format
    """

    def __init__(self, cutting_threshold = 0., **params):
        self.params = ParameterGrid(params)
        self.fitted = False
        self.cutting_threshold = cutting_threshold

    def fit(self, x, y, h, vsize = 0.15, val = None, random_state = 42, groups = None, **args):
        """
            This method is used to train an instance of multi layer perceptron

            Args:
                x (np.ndarray): A numpy array of the input features
                y (np.ndarray): A numpy array of the target label (1 dimensional for binary classification)
                vsize (float, optional): Percentage of data used for validation and dev. Defaults to 0.15.
                val ((np.ndarray, np.ndarray), optional): Tuple of validation data and labels. Defaults to None.
                l1_penalty (float, optional): L1 penalty used for the loss
                platt_calibration (bool, optional): Compute a platt calibration for the model (based on val set).
                random_state (int, optional): Random seed used for training and data split. Defaults to 100.

            Returns:
                self
        """
        # Preprocess data
        processed_data = self._preprocess_train_(x, y, h, vsize, val, groups, random_state)
        x_train, y_train, h_train, x_val, y_val, h_val, x_dev, y_dev, h_dev = processed_data

        # Find best model - Grid search
        self.torch_model, best_perf = None, np.inf
        for param in self.params:
            # Create and train model
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            model = self._gen_torch_model_(x_train.size(1), 1, param)
            model = train_defer(model, x_train, y_train, h_train, x_val, y_val, h_val, **args)
            perf = compute_defer_loss(model, x_dev, y_dev, h_dev).item()

            if perf < best_perf:
                best_perf = perf
                self.torch_model = model

        self.fitted = True
        return self

    def predict(self, x, h = None):
        """
            Estimates the predicted outcome for x

            Args:
                x (np.narray): A numpy array of the input features
                h (np.narray): A numpy array of the human decision

            Returns:
                np.narray: Predicted outcome
        """
        x = self._preprocess_(x)
        if self.fitted:
            output, deferal = self.torch_model(x)
            output, deferal = output.detach().numpy().flatten(), deferal.detach().numpy().flatten()
            output[deferal > 0.5] = np.nan if h is None else h[deferal > 0.5] # Must be replaced by human decision
            return output
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `predict`.")

    def _gen_torch_model_(self, inputdim, outputdim, param):
        assert outputdim == 1, "Multi class not handle at the moment"
        return DeferMultiLayerPerceptron_Torch(inputdim, outputdim, **param).double()

    def _preprocess_(self, x):
        x = x.values if (isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)) else x
        return torch.from_numpy(x).double()

    def _preprocess_train_(self, x, y, h, vsize, val, groups = None, random_state = 42):
        if groups is None:
            splitter = ShuffleSplit(n_splits = 1, test_size = vsize, random_state = random_state)
        else:
            splitter = GroupShuffleSplit(n_splits = 1, test_size = vsize, random_state = random_state)
        
        x_train = self._preprocess_(x)
        y_train = self._preprocess_(y)
        h_train = self._preprocess_(h)

        train, dev = next(splitter.split(x_train, y_train, groups))
        x_dev, y_dev, h_dev = x_train[dev], y_train[dev], h_train[dev]
        x_train, y_train, h_train = x_train[train], y_train[train], h_train[train]

        if val is None:
            train, val = next(splitter.split(x_train, y_train, None if groups is None else groups[train]))
            x_val, y_val, h_val = x_train[val], y_train[val], h_train[val]
            x_train, y_train, h_train = x_train[train], y_train[train], h_train[train]
        else:
            x_val, y_val, h_val = val
            x_val = self._preprocess_(x_val)
            y_val = self._preprocess_(y_val)
            h_val = self._preprocess_(h_val)

        return x_train, y_train, h_train, x_val, y_val, h_val, x_dev, y_dev, h_dev

class DeferMultiLayerPerceptron_Torch(nn.Module):

    def __init__(self, input_dim, output_dim = 1, layers = [], 
        activation = 'ReLU', dropout = 0.5):
        super(DeferMultiLayerPerceptron_Torch, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mlp = create_mlp(input_dim, layers + [output_dim], activation, dropout)
        self.defer = create_mlp(input_dim + output_dim, layers + [output_dim], activation, dropout)

    def forward(self, x):
        output = self.mlp(x)

        # Want independent tensor so remove gradient of output
        concatentation = torch.cat((output.clone().detach(), x), 1)
        deferal = self.defer(concatentation)

        return output, deferal

def compute_defer_loss(model, x, y, h):
    output, deferal = model(x)
    output, deferal = output.view(-1), deferal.view(-1)
    output = deferal * h + (1 - deferal) * output
    return nn.BCELoss()(output, y) 

def train_defer(model,
            x_train, y_train, h_train, x_valid, y_valid, h_valid,
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
            hb = h_train[index[j*bs:(j+1)*bs]]
            
            if xb.shape[0] == 0:
                continue

            optimizer.zero_grad()
            loss = compute_defer_loss(model, xb, yb, hb) 
            loss.backward()
            optimizer.step()

        model.eval()
        xb, yb, hb = x_valid, y_valid, h_valid
        valid_loss = compute_defer_loss(model, xb, yb, hb).item() 
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