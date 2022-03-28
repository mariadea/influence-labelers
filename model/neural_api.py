import torch
from torch.autograd.functional import hessian, jacobian
import numpy as np
import pandas as pd
from tqdm import tqdm

from .utils import train_mlp
from .loss import compute_loss
from .amalgamation import compute_influence
from .neural_torch import MultiLayerPerceptron_Torch

class BinaryMLP:
    """
        Model API to call for using the method
        Preprocess data to shape it to the right format
    """

    def __init__(self, **params):
        self.params = params
        self.fitted = False

    def fit(self, x, y, h, vsize = 0.15, val = None, l1_penalty = 0.001, random_state = 42, **args):
        """
            This method is used to train an instance of multi layer perceptron

            Args:
                x (np.ndarray): A numpy array of the input features
                y (np.ndarray): A numpy array of the target label (1 dimensional for binary classification)
                vsize (float, optional): Percentage of data used for validation. Ignored if val is provided. Defaults to 0.15.
                val ((np.ndarray, np.ndarray), optional): Tuple of validation data and labels. Defaults to None.
                l1_penalty (float, optional): L1 penalty used for the loss
                random_state (int, optional): Random seed used for training and data split. Defaults to 100.

            Returns:
                self
        """
        # Preprocess data
        processed_data = self._preprocess_train_(x, y, vsize, val, random_state)
        x_train, y_train, x_val, y_val = processed_data
        self.experts_training, self.x, self.y = h.values if isinstance(h, pd.Series) else h, self._preprocess_(x), self._preprocess_(y)
        self.experts = np.sort(np.unique(h))
        self.l1_penalty = l1_penalty # Save to ensure l1 penalyt consistent

        # Create and train model
        torch.manual_seed(random_state)
        model = self._gen_torch_model_(x_train.size(1), 1)
        model = train_mlp(model, x_train, y_train, x_val, y_val, l1_penalty = l1_penalty, **args)

        # Update model
        self.torch_model = model.eval()
        self.fitted = True

        # Estimate hessian of training loss
        theta = self.torch_model.get_last_weights() # Use the parameters of the last layer only
        try:
            self.hess = hessian(lambda weight: compute_loss(self.torch_model.replace_last_weights(weight), self.x, self.y, l1_penalty = l1_penalty), theta, create_graph = True).squeeze()
        except:
            raise ValueError('Architecture leads to singular weights matrix for last layer: Use another architecture or increase l1_penalty.')

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
            return self.torch_model(x).detach().numpy()
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

        # Remove null theta from hessian
        hess = self.hess[theta.squeeze() > 0, :][:, theta.squeeze() > 0]
        grad_p = grad_p[:, theta.squeeze() > 0]

        for i, expert in enumerate(self.experts):
            influence_matrix[i] = compute_influence(self.torch_model, grad_p, 
                self.x[self.experts_training == expert], self.y[self.experts_training == expert], 
                hess, l1_penalty = self.l1_penalty).detach()

        return influence_matrix

    def _gen_torch_model_(self, inputdim, outputdim):
        assert outputdim == 1, "Multi class not handle at the moment"
        model = MultiLayerPerceptron_Torch(inputdim, outputdim, **self.params).double()
        return model

    def _preprocess_(self, x):
        x = x.values if (isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)) else x
        return torch.from_numpy(x).double()

    def _preprocess_train_(self, x, y, vsize, val, random_state):
        np.random.seed(random_state)

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        x = self._preprocess_(x)
        y = self._preprocess_(y)

        x_train, y_train = x[indices], y[indices]

        if val is None:
            vsize = int(vsize * x_train.shape[0])
            x_val, y_val = x_train[-vsize:], y_train[-vsize:]
            x_train, y_train = x_train[:-vsize], y_train[:-vsize]
        else:
            x_val, y_val = val
            x_val = self._preprocess_(x_val)
            y_val = self._preprocess_(y_val)

        return x_train, y_train, x_val, y_val
