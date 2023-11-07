import argparse


def create_parser():
    parser = argparse.ArgumentParser(description="Trains NLP model on Reddit posts.")
    return parser


def add_positional_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "model",
        metavar="Model",
        type=str,
        help="Model to train. Options: MLP, RNN, LSTM, BERT_MLP, BERT_RNN, BERT_LSTM.",
    )
    # Add split variable, as I don't want to change this.
    parser.set_defaults(split=[0.7, 0.15, 0.15])

    return parser


def add_optional_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    parser.add_argument(
        "--sample_rate",
        metavar="Sample Rate",
        type=float,
        help="% of dataset sampled in model.",
        nargs="?",  # optional argument (0 or 1 args)
        default=0.1,
    )
    parser.add_argument(
        "--epochs",
        metavar="Training Epochs",
        type=int,
        help="Number of epochs to train model for.",
        nargs="?",
        default=30,
    )
    parser.add_argument(
        "--max_features",
        metavar="Max Features",
        type=int,
        help="Number of features to use in vectorisation for the word embedding models.",
        nargs="?",
        default=2**16,
    )
    parser.add_argument(
        "--posts_length",
        metavar="Maximum posts per author",
        type=int,
        help="Maximum number of posts per author in vectorisation for word embedding models.",
        nargs="?",
        default=100,
    )
    parser.add_argument(
        "--sentences_length",
        metavar="Sentence Length",
        type=int,
        help="Length of sentences to use in vectorisation for the sentence embedding models.",
        nargs="?",
        default=150,
    )
    parser.add_argument(
        "--transformer_model",
        metavar="Transformer Model",
        type=str,
        help="Sentence Transformer model to use for sentence embedding models.",
        nargs="?",
        default="sentence-transformers/all-MiniLM-L6-v2",
    )

    return parser
