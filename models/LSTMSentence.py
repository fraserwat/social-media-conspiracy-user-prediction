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
import torch.nn.utils.rnn as rnn_utils
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
        authors_lengths = []

        # Process each author's posts
        for author_posts in authors:
            post_embeddings = self.sentence_transformer.encode(
                author_posts, convert_to_tensor=True, show_progress_bar=False
            )
            # Now posts_embeddings is a 2D tensor of shape [num_posts, embedding_dim]
            authors_embeddings.append(post_embeddings)
            # Each author has a different number of posts, so we keep lengths of each post
            authors_lengths.append(post_embeddings.shape[0])

        # Pad sequences so each author has the same number of posts (some are padded with 0s)
        authors_padded = rnn_utils.pad_sequence(authors_embeddings, batch_first=True)

        # Convert list of lengths to a tensor
        lengths_tensor = torch.tensor(
            authors_lengths, dtype=torch.int64, device=authors_padded.device
        )

        # Pack the padded sequences into a PackedSequence object
        packed_input = rnn_utils.pack_padded_sequence(
            authors_padded, lengths_tensor, batch_first=True, enforce_sorted=False
        )

        # Pass the packed sequence through the first LSTM layer
        lstm_posts_output, (hidden_state, cell_state) = self.lstm_posts(packed_input)

        # In a many-to-one approach, we're only interested in the last output of LSTM sequence.
        # We can use the lengths_tensor to select the last output for each sequence.
        lstm_posts_output_unpacked, _ = rnn_utils.pad_packed_sequence(
            lstm_posts_output, batch_first=True
        )
        last_outputs = lstm_posts_output_unpacked[
            torch.arange(lstm_posts_output_unpacked.size(0)), lengths_tensor - 1
        ]

        # Pass author sequences through the second LSTM layer
        lstm_authors_output, _ = self.lstm_authors(last_outputs)

        # Take output from the last LSTM layer for prediction, removing sequence length dimension
        final_output = lstm_authors_output.squeeze(1)

        # Apply dropout and pass through the linear layer with sigmoid activation
        final_output = F.dropout(
            final_output, p=self.dropout_rate, training=self.training
        )
        final_output = torch.sigmoid(self.output(final_output))

        return final_output
