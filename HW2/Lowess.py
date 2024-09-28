import numpy as np
from sklearn.linear_model import LinearRegression as lm
from sklearn.utils.validation import check_is_fitted


class Lowess:
    def __init__(self, kernel: str, tau: float):
        """
        Initialize Lowess model with kernel selection and tau value.

        Args:
            kernel: The algorithm that will perform pattern analysis on data
            tau: Measures rank correlation between features and target data
        """
        self.kernel = kernel
        self.tau = tau

    
    def fit(self, X, y):
        """
        Fits the Lowess model to the training data.

        Args:
            X_train: Training features (numpy array or dataframe)
            y_train: Training target values (numpy array or series)
        """
        self.X_train = np.asarray(X) # set features to np.array
        self.y_train = np.asarray(y) # set target to np.array

        self.is_fitted = True

    def _compute_weights(self, x_new, x):
        """
        Compute weights using the kernel function and tau (bandwidth).
        """
        return self.kernel((x - x_new) / self.tau)
    
    def predict(self, x_new):
        check_is_fitted(self)
        X = self.X_train
        y = self.y_train

        if np.isscalar(x_new):
            x_new = np.array([x_new])

        y_pred = np.zeros(len(x_new))

        for i, x_n in enumerate(x_new):
            weights = self._compute_weights(x_n, X)
        if np.sum(weights) == 0:
            y_pred[i] = np.nan  # Handle case where all weights are zero
        else:
            # Compute the weighted average
            y_pred[i] = np.sum(weights * y) / np.sum(weights)

        return y_pred

    