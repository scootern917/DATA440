import numpy as np
from sklearn.base import clone

class GradientBooster:
    def __init__(self, model: str, resmodel: str, n_boosting_steps: int):
        """
        Initialize the Gradient Booster with a main model, a residual model, and boosting steps.
        
        Args:
            model: The primary model used for initial prediction
            resmodel: The residual model used to fit the residuals at each boosting step
            n_boosting_steps: The number of boosting steps (iterations) to perform
        """
        self.model = model  # main model (will end up being Lowess model)
        self.resmodel = resmodel # residual model (user's choice)
        self.n_boosting_steps = n_boosting_steps # choose number of boosting steps
        self.is_fitted = False  # check to see if model is fitted, default as False

        self.residuals = [] # empty list to store residual model results
    

    def fit(self, X_train, y_train): # fit the train data for the GradientBoost model
        """
        Fit the GradientBooster to the training data.
        
        Args:
            X_train: Training features (numpy array or dataframe)
            y_train: Training target values (numpy array or series)
        """

        self.model.fit(X_train, y_train) # Fit the main model to the training data

        y_pred = self.predict(X_train) # Get the initial predictions from the main model

        for n in range(self.n_boosting_steps): # Iteratively fit residuals with boosting steps

            # Calculate residuals (difference between actual target and predicted)
            residuals = y_train - y_pred
            res_model = clone(self.resmodel)

            res_model.fit(X_train, residuals) # Fit the residual model on the residuals

            self.residuals.append(res_model) # Store the fitted residual model

            # Update predictions by adding the residual model's predictions
            y_pred = y_pred + res_model.predict(X_train)

        self.is_fitted = True # Mark the model as fitted



    def predict(self, X_test):
        """
        Predict target values for the test data using the fitted GradientBooster.
        
        Args:
            X_test: Test features (numpy array or dataframe)
        
        Returns:
            Final predictions after boosting (numpy array)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Please call 'fit' before predicting.")


        y_pred = self.model.predict(X_test) # Get initial predictions from the main model

        for res_model in self.residuals:
            y_pred = y_pred + res_model.predict(X_test) # Add the contribution from each residual model

        return y_pred
    


# I have no idea what these functions do but I couldn't solve a few errors and this is what I found online
    def get_params(self, deep=True):
        return {
        'model': self.model,
        'resmodel': self.resmodel,
        'n_boosting_steps': self.n_boosting_steps
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self
