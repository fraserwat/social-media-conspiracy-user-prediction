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

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F


class SentenceLSTM(torch.nn.Module):
    def __init__(
        self, bert_model_path: str, hidden_size: int = 32, dropout_rate: float = 0.0
    ):
        super().__init__()

        # Initialize the Sentence Transformer for embedding sentences
        self.sentence_transformer = SentenceTransformer(bert_model_path)

        # Define the LSTM components
        # First LSTM layer processes the embeddings of the sentences within a post
        self.lstm_posts = torch.nn.LSTM(
            input_size=self.sentence_transformer.get_sentence_embedding_dimension(),
            hidden_size=hidden_size,
            batch_first=True,
            # dropout=dropout_rate,
            num_layers=1,
        )
        # Second LSTM layer processes the final **sequence of outputs from the first LSTM layer**
        # In theory, this should capture the dependencies between different posts of an author
        self.lstm_authors = torch.nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
            # dropout=dropout_rate,
            num_layers=1,
        )
        self.dropout_rate = dropout_rate
        self.output = torch.nn.Linear(hidden_size, 1)

    def forward(self, authors):
        authors_embeddings = []

        # Process each author's posts
        for author_posts in authors:
            posts_embeddings = self.sentence_transformer.encode(
                author_posts, convert_to_tensor=True, show_progress_bar=False
            )

            # Pass each post through the first LSTM
            lstm_posts_output, _ = self.lstm_posts(posts_embeddings)

            # Collect the last output for each post sequence

            print(self.lstm_posts(posts_embeddings))
            print(len(self.lstm_posts(posts_embeddings)))

            last_posts_output = lstm_posts_output[:, -1, :]
            authors_embeddings.append(last_posts_output)

        # Pad the sequences to have the same length
        authors_padded = pad_sequence(authors_embeddings, batch_first=True)

        # Process the author sequences through the second LSTM
        lstm_authors_output, _ = self.lstm_authors(authors_padded)

        # Use the output of the last timestep
        final_output = lstm_authors_output[:, -1, :]

        # Apply dropout and pass through the linear layer with sigmoid activation
        final_output = F.dropout(
            final_output, p=self.dropout_rate, training=self.training
        )
        final_output = torch.sigmoid(self.output(final_output))

        return final_output
