from LowessClass import Lowess
import numpy as np
from sklearn.linear_model import LogisticRegression as LogReg

def expit(x):
    # computing the sigmoid function
    return 1 / (1 + np.exp(-x))

class LocallyWeightedLogisticRegression:
    def __init__(self, kernel, lr=0.01, tau=0.05):
        # initializing locally weighted logistic regression model
        self.lr = lr   # learning rate
        self.tau = tau  # bandwidth parameter
        self.lowess = Lowess(kernel=kernel, tau=tau)  # initializing the lowess model
        # creating dictionary to store separate logistic regression models for each class in One-vs-Rest approach
        self.models = {}

    def fit(self, xtrain, ytrain):
        self.lowess.fit(xtrain, ytrain) # fitting the lowess model on training data
        self.xtrain = xtrain  # storing training features
        self.ytrain = ytrain  # storing training classifcations
        self.classes = np.unique(ytrain)  # extracting unique class labels from the training target (one for each classification)

        # training a model for each class (one v rest)
        for cls in self.classes:
            binary_ytrain = (ytrain == cls).astype(int)  # creating binary labels for current class
            model = LogReg(multi_class='ovr') # initializing the logistic regression model with ovr (one vs rest)
            model.fit(self.xtrain, binary_ytrain)  # fitting model for current class
            self.models[cls] = model # storing trained model in dictionary

    def predict(self, xtest):
        ypred = []   # initializing a list to store predictions

        for i in range(len(xtest)):  # iterating over each test sample
            class_probabilities = [] # initializing list to store probabilities for each class

            # making predictions for each class
            for cls in self.classes:
                model = self.models[cls]  # retrieving model for current class
                prob = model.predict_proba(xtest[i].reshape(1, -1))[:, 1]  # getting predicted probability for current class
                class_probabilities.append(prob) # appending probability to list

            # choosing class with the highest predicted probability
            ypred.append(self.classes[np.argmax(class_probabilities)]) # appending the predicted class

        # returning predictions as a NumPy array
        return np.array(ypred)