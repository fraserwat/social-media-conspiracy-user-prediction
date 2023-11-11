"""
Due to the sequential structure of the post data, we built a many to one RNN model as a second
baseline. The input is a sequence of trained word or BERT sentence embeddings for the author's
entire post history, which we fed to a simple RNN layer with 64 hidden units. As with the MLP, the
RNN applies a sigmoid activation on the output layer. It also uses the same cost function, Adam
optimizer settings, and regularization strategies.
"""

import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class RNN(torch.nn.Module):
    def __init__(
        self,
        bert_model_path: str | None = None,
        input_size: int = 0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # Initialize the Sentence Transformer if a BERT model path is provided
        self.sentence_transformer = None
        if bert_model_path:
            self.sentence_transformer = SentenceTransformer(bert_model_path)
            self.input_size = (
                self.sentence_transformer.get_sentence_embedding_dimension()
            )
        else:
            self.input_size = input_size

        # Define the RNN components
        self.rnn = torch.nn.RNN(
            input_size=self.input_size, hidden_size=64, batch_first=True
        )
        self.output = torch.nn.Linear(64, 1)
        self.dropout_rate = dropout_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, author_posts):
        # Embeddings
        if self.sentence_transformer:
            assert [text for text in author_posts if text], "Batch is empty."
            author_embeddings = []
            lengths = []
            for posts in author_posts:
                post_embeddings = self.sentence_transformer.encode(
                    posts, convert_to_tensor=True, show_progress_bar=False
                )
                author_embeddings.append(post_embeddings)
                # Stores length of each sequence for padding
                lengths.append(post_embeddings.shape[0])

            author_embeddings = [
                embedding.to(self.device) for embedding in author_embeddings
            ]
            # Pad the sequences to have the same length
            x_padded = pad_sequence(author_embeddings, batch_first=True)
        else:
            # If we're not using SBERT embeddings we're using word embeddings.
            x_padded = pad_sequence(author_posts, batch_first=True)
        x_padded = x_padded.to(self.device)
        # Running the RNN model.
        # If we're using SBERT embeddings, we have some extra steps with the packed sequences
        if self.sentence_transformer:
            # Convert lengths to a tensor
            lengths_tensor = torch.tensor(
                lengths, dtype=torch.int64, device=x_padded.device
            )

            # Pack the padded sequences
            x_packed = pack_padded_sequence(
                x_padded, lengths_tensor, batch_first=True, enforce_sorted=False
            )

            # Process the sequence through the RNN
            rnn_output, hidden_state = self.rnn(x_packed)

            # As we're only using the last output, we can just use take the hidden_state
            # hidden_state is a tensor of shape [num_layers * num_directions, batch, hidden_size]
            # If batch_first=True, we need to transpose it to bring the batch dimension to dim 0
            x = hidden_state.transpose(0, 1).contiguous().view(-1, self.rnn.hidden_size)
        else:
            # Process the sequence through the RNN
            x, _ = self.rnn(x_padded)

        # Apply dropout
        x = F.dropout(
            x,
            p=self.dropout_rate,
            training=self.training,
        )
        # Pass the final output through the linear layer and apply sigmoid
        x = torch.sigmoid(self.output(x))
        return x
