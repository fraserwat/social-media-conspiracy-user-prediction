"""
Hoping to capture more long-term temporal dependencies, we built a many to one LSTM model. For
the trained word embeddings, we used 2 hidden LSTM layers, one with 32 hidden units outputting a
sequence and one with 32 hidden units. For the sentence embeddings, we also used 2 hidden LSTM
layers. We took the ordered sentence chunk encodings per post and passed each of them to a LSTM
with 32 hidden units. We then took the output of each of these LSTMs and combined them into a
new sequence that we then passed to a second LSTM also with 32 hidden units. We used the same
sigmoid activation on the output layer, cost function, Adam optimizer, and regularization
strategies as the baseline models.
"""

import torch.nn as nn


class BaselineLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim=32, output_dim=1, dropout_rate=0.0):
        super(BaselineLSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.lstm_batchnorm = nn.BatchNorm1d(hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.lstm_batchnorm(out)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        # TODO: Logic to put this back in (& replace next line) if nth_dimension > 2
        # out = self.linear(out[:, -1, :])  # Many-to-One, taking the last output
        out = self.lstm_batchnorm(out)
        out = self.dropout(out)
        out = self.linear(out)
        out = self.sigmoid(out)
        return out
