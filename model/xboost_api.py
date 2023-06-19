import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ParameterGrid

from .neural_api import BinaryMLP
from .loss import compute_loss

class BinaryXGB(BinaryMLP):
    """
        Model API to call for using a xgboost model
    """

    def fit(self, x, y, h, vsize = 0.15, val = None, l1_penalty = 0., platt_calibration = False, random_state = 42, groups = None, **args):
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

        # Find best model - Grid search
        self.model, best_perf = None, np.inf
        for param in self.params:
            # Create and train model
            np.random.seed(random_state)
            model = xgb.XGBClassifier(random_state = random_state, use_label_encoder = False, **param)
            model = model.fit(x_train, y_train, **args)
            perf = -model.score(x_dev, y_dev) # Best score is 1 (values csan go negative) 
            
            if perf < best_perf:
                best_perf = perf
                self.model = model

        if self.model is None:
            raise ValueError('Architecture leads to singular weights matrix for last layer: Use another architecture or increase l1_penalty.')

        self.fitted = True
        self.calibrated = False

        # Compute calibration
        if platt_calibration:
            # Calibrate NN on validation set - Platt
            pred_val = self.model.predict_proba(x_val)
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
            out_xg = self.model.predict_proba(x)
            return self.calibration.predict_proba(out_xg)[:, 1] if self.calibrated else out_xg[:, 1].flatten()
        else:
            raise Exception("The model has not been fitted yet. Please fit the " +
                            "model using the `fit` method on some training data " +
                            "before calling `predict`.")

    def influence(self, x, batch = 1000):
        raise NameError('Computing influence with xgboost is not implemented')
    
    def _preprocess_(self, x):
        return x.values if (isinstance(x, pd.DataFrame) or isinstance(x, pd.Series)) else x