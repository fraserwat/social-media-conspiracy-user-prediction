"""
Our first baseline model is a simple feedforward neural network with 2 dense hidden layers and a
single unit output layer. We use log loss as our cost function and the Adam optimizer with default
hyperparameter settings. At each layer we reduce dimensionality, with 256 units at the first hidden
layer, and 128 units at the second hidden layer. We use ReLU activation at each hidden layer. In an
attempt to mitigate overfitting we use L2 regularization at each hidden layer followed by batchnorm,
as well as early stopping (num epochs=30, patience=3). We apply a sigmoid transform to the output
to convert our predictions to probability space
"""

import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, input_size, dropout_rate=0.0):
        # This line ensures that the superclass (torch.nn.Module) is called and initialised.
        super().__init__()

        self.linear1 = torch.nn.Linear(input_size, 256)
        self.linear1_batchnorm = torch.nn.BatchNorm1d(256)

        self.linear2 = torch.nn.Linear(256, 128)
        self.linear2_batchnorm = torch.nn.BatchNorm1d(128)

        self.output = torch.nn.Linear(128, 1)
        self.activation = torch.nn.ReLU()

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear1_batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.linear2(x)
        x = self.linear2_batchnorm(x)
        x = self.activation(x)
        x = self.dropout(x)

        x = self.output(x)

        return torch.sigmoid(x)
