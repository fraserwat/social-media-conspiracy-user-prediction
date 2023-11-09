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
from sentence_transformers import SentenceTransformer


class MLP(torch.nn.Module):
    def __init__(
        self,
        bert_model_path: str | None = None,
        input_size: int = 0,
        dropout_rate: float = 0.0,
    ):
        super().__init__()

        # if bert_model_path is provided, use Sentence Transformer to get sentence embeddings
        # otherwise, sentence_transformer = None and we use word embeddings. Input size is either
        # passed in as an argument (word model) or calculated from the sentence transformer model.
        self.sentence_transformer, self.input_size = None, input_size
        if bert_model_path:
            # Initialize the Sentence Transformer
            self.sentence_transformer = SentenceTransformer(bert_model_path)
            self.input_size = (
                self.sentence_transformer.get_sentence_embedding_dimension()
            )

        # Define the MLP components
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 1),
        )

    def forward(self, author_posts):
        # This only runs if BERT is used
        if self.sentence_transformer:
            # Check the input is not empty (i.e. batch contains at least one non-empty text).
            assert [text for text in author_posts if text], "Batch is empty."
            # Process each author's posts
            author_embeddings = []
            for posts in author_posts:
                # Encode posts into embeddings
                post_embeddings = self.sentence_transformer.encode(
                    posts, convert_to_tensor=True, show_progress_bar=False
                )
                # Average pooling across posts for a single author
                embedding = torch.mean(post_embeddings, dim=0)
                author_embeddings.append(embedding)

            # Stack posts to create a batch tensor
            x = torch.stack(author_embeddings)

        else:
            x = author_posts
        return torch.sigmoid(self.mlp(x))
