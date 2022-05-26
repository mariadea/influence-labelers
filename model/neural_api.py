import torch
from torch.autograd.functional import hessian, jacobian
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

from .utils import train_mlp
from .loss import compute_loss
from .amalgamation import compute_influence
from .neural_torch import MultiLayerPerceptron_Torch

class BinaryMLP:
    """
        Model API to call for using the method
        Preprocess data to shape it to the right format
    """

    def __init__(self, cutting_threshold = 0., **params):
        self.params = ParameterGrid(params)
        self.fitted = False
        self.cutting_threshold = cutting_threshold

    def fit(self, x, y, h, vsize = 0.15, val = None, l1_penalty = 0., platt_calibration = False, random_state = 42, check = False, groups = None, **args):
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
        processed_data = self._preprocess_train_(x, y, vsize, val, groups, random_state)
        x_train, y_train, x_val, y_val, x_dev, y_dev = processed_data
        self.experts_training, self.x, self.y = h.values if isinstance(h, pd.Series) else h, self._preprocess_(x), self._preprocess_(y)
        self.experts = np.sort(np.unique(h))
        self.l1_penalty = l1_penalty # Save to ensure l1 penalyt consistent

        # Find best model - Grid search
        self.torch_model, best_perf = None, np.inf
        for param in self.params:
            # Create and train model
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            model = self._gen_torch_model_(x_train.size(1), 1, param)
            model = train_mlp(model, x_train, y_train, x_val, y_val, l1_penalty = l1_penalty, **args)
            perf = compute_loss(model, x_dev, y_dev, l1_penalty).item()
            
            if check:
                # Estimate hessian of training loss
                try:
                    theta = model.get_last_weights() # Use the parameters of the last layer only
                    hess = hessian(lambda weight: compute_loss(model.replace_last_weights(weight), self.x, self.y, l1_penalty = l1_penalty), theta, create_graph = True).squeeze()
                    hess = hess[theta.abs().squeeze() > self.cutting_threshold, :][:, theta.abs().squeeze() > self.cutting_threshold]
                    torch.inverse(hess)
                except:
                    # Ignore maybe next is invertible
                    continue

            if perf < best_perf:
                best_perf = perf
                self.torch_model = model
                self.hess = hess if check else None

        if self.torch_model is None:
            raise ValueError('Architecture leads to singular weights matrix for last layer: Use another architecture or increase l1_penalty.')

        self.fitted = True
        self.calibrated = False

        # Compute calibration
        if platt_calibration:
            # Calibrate NN on validation set - Platt
            pred_val = self.torch_model(x_val).detach().numpy()
            self.calibration = LogisticRegression(random_state = random_state).fit(pred_val, y_val)
            self.calibrated = True

        return self

    def predict(self, x):
        """
            Estimates the predicted outcome for x

            Args:
                x (np.narray): A numpy array of the input features

            Returns:
                np.narray: Predicted outcome
        """
        x = self._preprocess_(x)
        if self.fitted:
            out_nn = self.torch_model(x).detach().numpy()
            return self.calibration.predict_proba(out_nn)[:, 1] if self.calibrated else out_nn.flatten()
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `predict`.")

    def influence(self, x, batch = 1000):
        """
            Computes the influence of experts

            Args:
                x (np.narray): A numpy array of the input features

            Returns:
                np.narray: A 1d array of the influence of each expert
        """ 
        assert self.hess is not None, "Model hasn't been trained for influence"
        if batch is not None:
            # Batch computation to avoid too large matrix with hess shared
            batched_inf = []
            for i in range(len(x) // batch + 1):
                if len(x[i * batch:(i+1) * batch]) > 0:
                    batched_inf.append(self.influence(x[i * batch:(i+1) * batch], batch = None))
            return np.concatenate(batched_inf, axis = 1)
        
        theta = self.torch_model.get_last_weights()
        x = self._preprocess_(x)
        influence_matrix = np.zeros((self.experts.shape[0], x.shape[0]))

        grad_p = jacobian(lambda weight: self.torch_model.replace_last_weights(weight)(x), theta, create_graph = True).squeeze()

        # Remove null theta
        grad_p = grad_p[:, theta.abs().squeeze() > self.cutting_threshold]

        for i, expert in enumerate(self.experts):
            influence_matrix[i] = compute_influence(self.torch_model, grad_p, 
                self.x[self.experts_training == expert], self.y[self.experts_training == expert], 
                self.hess, l1_penalty = self.l1_penalty, cutting_threshold = self.cutting_threshold).detach()

        return influence_matrix

    def _gen_torch_model_(self, inputdim, outputdim, param):
        assert outputdim == 1, "Multi class not handle at the moment"
        model = MultiLayerPerceptron_Torch(inputdim, outputdim, **param).double()
        return model

    def _preprocess_(self, x):
        x = x.values if (isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)) else x
        return torch.from_numpy(x).double()

    def _preprocess_train_(self, x, y, vsize, val, groups = None, random_state = 42):
        if groups is None:
            splitter = ShuffleSplit(n_splits = 1, test_size = vsize, random_state = random_state)
        else:
            splitter = GroupShuffleSplit(n_splits = 1, test_size = vsize, random_state = random_state)
        
        x_train = self._preprocess_(x)
        y_train = self._preprocess_(y)

        train, dev = next(splitter.split(x_train, y_train, groups))
        x_dev, y_dev = x_train[dev], y_train[dev]
        x_train, y_train = x_train[train], y_train[train]

        if val is None:
            train, val = next(splitter.split(x_train, y_train, None if groups is None else groups[train]))
            x_val, y_val = x_train[val], y_train[val]
            x_train, y_train = x_train[train], y_train[train]
        else:
            x_val, y_val = val
            x_val = self._preprocess_(x_val)
            y_val = self._preprocess_(y_val)

        return x_train, y_train, x_val, y_val, x_dev, y_dev