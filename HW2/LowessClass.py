import numpy as np
import sklearn.linear_model as lm
from sklearn.utils.validation import check_is_fitted
from scipy.spatial.distance import cdist

class Lowess:
    def __init__(self, kernel=None, tau=0.5):
        """
        Initialize Lowess model with kernel selection and tau value.

        Args:
            kernel: The algorithm that will perform pattern analysis on data
            tau: Measures rank correlation between features and target data
        """
        self.kernel = kernel
        self.tau = tau

    def fit(self, x, y):
        """
        Fits the Lowess model to the training data.

        Args:
            xtrain: Training features (numpy array or dataframe)
            ytrain: Training target values (numpy array or series)
        """
        self.xtrain = np.asarray(x) # Store training data
        self.ytrain = np.asarray(y) # Store target values
        self.is_fitted_ = True

    
    def predict(self, x_new):
        """
        Predicts target values.
        """
        check_is_fitted(self)
        X = self.xtrain
        y = self.ytrain

        lin_reg = lm.Ridge(alpha=0.001, fit_intercept=True)
        distances = cdist(X,x_new,metric='Euclidean')

        w = self.kernel(distances / (2 * self.tau))

        n = x_new.shape[0]
        yest_test = np.zeros(n)

        #Looping through all x-points
        for i in range(n):
            lin_reg.fit(np.diag(w[:,i]) @ X, np.diag(w[:,i]) @ y)
            yest_test[i] = lin_reg.predict(x_new[i].reshape(1,-1)).squeeze()
        return yest_test

    
