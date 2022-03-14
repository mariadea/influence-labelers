import torch
from torch.autograd.functional import hessian
import numpy as np
from tqdm import tqdm

from .loss import *
from .utils import train_mlp
from .neural_torch import MultiLayerPerceptron_Torch

class BinaryMLP:
    """
        Model API to call for using the method
        Preprocess data to shape it to the right format
    """

    def __init__(self, **params):
        self.params = params
        self.fitted = False

    def fit(self, x, y, h, vsize = 0.15, val = None, random_state = 100, **args):
        """
            This method is used to train an instance of multi layer perceptron

            Args:
                x (np.ndarray): A numpy array of the input features
                y (np.ndarray): A numpy array of the target label (1 dimensional for binary classification)
                vsize (float, optional): Percentage of data used for validation. Ignored if val is provided. Defaults to 0.15.
                val ((np.ndarray, np.ndarray), optional): Tuple of validation data and labels. Defaults to None.
                random_state (int, optional): Random seed used for training and data split. Defaults to 100.

            Returns:
                self
        """
        # Preprocess data
        processed_data = self._preprocess_train_(x, y, vsize, val, random_state)
        x_train, y_train, x_val, y_val = processed_data
        self.experts_training, self.x, self.y = h, self._preprocess_(x), self._preprocess_(y)
        self.experts = np.unique(h)

        # Create and train model
        torch.manual_seed(random_state)
        model = self._gen_torch_model_(x_train.size(1), 1)
        model = train_mlp(model, x_train, y_train, x_val,y_val, **args)

        # Update model
        self.torch_model = model.eval()
        self.fitted = True
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

    def influence(self, x):
        """
            Computes the influence of experts

            Args:
                x (np.narray): A numpy array of the input features

            Returns:
                np.narray: A 1d array of the influence of each expert
        """
        x = self._preprocess_(x)
        influence_matrix = np.zeros((self.experts.shape[0], x.shape[0]))
        
        # Estimate hessian of training loss
        theta = self.torch_model.get_last_weights() # Use the parameters of the last layer only
        hess = hessian(lambda weight: compute_loss(self.torch_model.replace_last_weights(weight), self.x, self.y), theta, create_graph = True).squeeze()

        grad_p = jacobian(lambda weight: self.torch_model.replace_last_weights(weight)(x), theta, create_graph = True).squeeze()

        # Remove null theta from hessian
        hess = hess[theta.squeeze() > 0, :][:, theta.squeeze() > 0]
        grad_p = grad_p[:, theta.squeeze() > 0]

        for expert in self.experts:
            influence_matrix[expert] = compute_influence(self.torch_model, grad_p, self.x[self.experts_training == expert], self.y[self.experts_training == expert], hess).detach()

        return influence_matrix


    def _gen_torch_model_(self, inputdim, outputdim):
        assert outputdim == 1, "Multi class not handle at the moment"
        model = MultiLayerPerceptron_Torch(inputdim, outputdim, **self.params).double()
        return model

    def _preprocess_(self, x):
        return torch.from_numpy(x).double()

    def _preprocess_train_(self, x, y, vsize, val, random_state):
        np.random.seed(random_state)

        indices = np.arange(x.shape[0])
        np.random.shuffle(indices)

        x_train, y_train = x[indices], y[indices]

        x_train = self._preprocess_(x_train)
        y_train = self._preprocess_(y_train)

        if val is None:
            vsize = int(vsize * x_train.shape[0])
            x_val, y_val = x_train[-vsize:], y_train[-vsize:]
            x_train, y_train = x_train[:-vsize], y_train[:-vsize]
        else:
            x_val, y_val = val
            x_val = self._preprocess_(x_val)
            y_val = self._preprocess_(y_val)

        return x_train, y_train, x_val, y_val
