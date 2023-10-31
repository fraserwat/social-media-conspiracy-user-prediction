"""
Due to the sequential structure of the post data, we built a many to one RNN model as a second
baseline. The input is a sequence of trained word or BERT sentence embeddings for the author's
entire post history, which we fed to a simple RNN layer with 64 hidden units. As with the MLP, the
RNN applies a sigmoid activation on the output layer. It also uses the same cost function, Adam
optimizer settings, and regularization strategies.
"""

import torch


class RNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size: int = 64, dropout: float = 0.05):
        super().__init__()

        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        # unpacking the RNN output and hidden state
        x, _ = self.rnn(x)
        x = self.output(x)

        return torch.sigmoid(x)
