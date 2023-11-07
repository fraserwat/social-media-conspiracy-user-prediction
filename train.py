import logging
from training import grid_search
from params import argparse_config as argparse

logging.basicConfig(level=logging.INFO)

# Getting arguments for training process
parser = argparse.create_parser()
parser = argparse.add_positional_arguments(parser)
parser = argparse.add_optional_arguments(parser)
args = parser.parse_args()


BERT_DIR, POS_DIR, NEG_DIR = (
    "data/bert.csv.gz",
    "data/non-q-posts-v2.csv.gz",
    "data/q-posts-v2.csv.gz",
)

grid_search_params = {
    "l2_penalty_weight": [0.001, 0.01, 0.1],
    "learning_rate": [0.0001, 0.001],
    "batch_size": [16, 32],
    "dropout_rate": [0.1, 0.25],
}

best_model = grid_search.perform(
    static_params=args,
    grid_search_params=grid_search_params,
    pos_path=POS_DIR,
    neg_path=NEG_DIR,
    bert_path=BERT_DIR,
)

logging.critical("BEST %s:\n%s", args.model, best_model)
