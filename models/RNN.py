"""
Due to the sequential structure of the post data, we built a many to one RNN model as a second
baseline. The input is a sequence of trained word or BERT sentence embeddings for the author's
entire post history, which we fed to a simple RNN layer with 64 hidden units. As with the MLP, the
RNN applies a sigmoid activation on the output layer. It also uses the same cost function, Adam
optimizer settings, and regularization strategies.
"""

import torch


class RNN(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=64)
        self.rnn_batchnorm = torch.nn.BatchNorm1d(64)
        self.output = torch.nn.Linear(64, 1)

    def forward(self, x):
        # unpacking the RNN output and hidden state
        x, hidden_state = self.rnn(x)

        # we want the last output / final state from the sequence of outputs
        # x is in the shape (batch_size, sequence_length, hidden_size)
        x = self.rnn_batchnorm(x[:, -1, :])
        x = self.output(x)

        return torch.sigmoid(x)
