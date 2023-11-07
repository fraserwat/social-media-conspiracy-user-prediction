"""
Due to the sequential structure of the post data, we built a many to one RNN model as a second
baseline. The input is a sequence of trained word or BERT sentence embeddings for the author's
entire post history, which we fed to a simple RNN layer with 64 hidden units. As with the MLP, the
RNN applies a sigmoid activation on the output layer. It also uses the same cost function, Adam
optimizer settings, and regularization strategies.
"""

import torch
from sentence_transformers import SentenceTransformer
from torch.nn.utils.rnn import pad_sequence


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
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, author_posts):
        # This only runs if BERT is used to get embeddings
        if self.sentence_transformer:
            assert [text for text in author_posts if text], "Batch is empty."
            author_embeddings = []
            for posts in author_posts:
                post_embeddings = self.sentence_transformer.encode(
                    posts, convert_to_tensor=True, show_progress_bar=False
                )
                author_embeddings.append(post_embeddings)

            # Pad the sequences to have the same length
            x_padded = pad_sequence(author_embeddings, batch_first=True)

        else:
            x_padded = pad_sequence(author_posts, batch_first=True)

        # Process the sequence through the RNN
        x, _ = self.rnn(x_padded)
        # Apply dropout to the last output of the sequence (N/A if word embeddings)
        x = self.dropout(x[:, -1, :] if self.sentence_transformer else x)
        # Pass the final output through the linear layer and apply sigmoid
        x = torch.sigmoid(self.output(x))
        return x
