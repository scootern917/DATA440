# pip3 install torch torchvision torchaudio
# !pip install ignite

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as torchfunctional

class SCADLinear(nn.Module):
    def __init__(self, input_size, lambda_val, a_val):
        super(SCADLinear, self).__init__() # initialize nn Module

        self.input_size = input_size
        self.lambda_val = lambda_val
        self.a_val = a_val
    
        self.linear = nn.Linear(input_size, 1, bias=True, device=torch.device('cpu'), dtype=torch.float64)

    def scad_derivative(self, beta_hat):
        deriv = self.lambda_val * ((beta_hat<=self.lambda_val) + (self.a_val*self.lambda_val - beta_hat) \
            * ((self.a_val*self.lambda_val - beta_hat)>0) / ((self.a_val-1)*self.lambda_val)*(beta_hat>self.lambda_val))
        return deriv


    def scad_penalty(self, beta_hat):
        abs_beta_hat = torch.abs(beta_hat)

        is_linear = abs_beta_hat <= self.lambda_val
        is_quadratic = torch.logical_and(self.lambda_val < abs_beta_hat, abs_beta_hat <= self.a_val * self.lambda_val)
        is_constant = (self.a_val * self.lambda_val) < abs_beta_hat
        
        linear_part = self.lambda_val * abs_beta_hat * is_linear
        quadratic_part = (2 * self.a_val * self.lambda_val * abs_beta_hat - beta_hat**2 - self.lambda_val**2) \
                / (2 * (self.a_val - 1)) * is_quadratic
        constant_part = (self.lambda_val**2 * (self.a_val + 1)) / 2 * is_constant
        return linear_part + quadratic_part + constant_part


    def forward(self, X):
        return self.linear(X) # make x compliant with linear model
    

    def loss(self, y_pred, y_true):
        mse = nn.MSELoss()(y_pred, y_true)
        penalty = torch.squeeze(self.scad_penalty(beta_hat=self.linear.weight))[1]

        return mse + penalty


    def fit(self, X, y, num_epochs=100, learning_rate=0.01):
        """
        Fit the ElasticNet model to the training data.

        Args:
            X (Tensor): Input data with shape (num_samples, input_size).
            y (Tensor): Target values with shape (num_samples, 1).
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for optimization.

        """
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        for epoch in range(num_epochs):
            self.train()
            optimizer.zero_grad()
            y_pred = self(X)
            loss = self.loss(y_pred, y)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 1000 == 0:
                print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}")
        
    def predict(self, X):
        self.eval()
        with torch.no_grad(): # forces custom gradient
            y_pred = self(X)
        return y_pred
    

    def get_coefficients(self):
        return self.linear.weight