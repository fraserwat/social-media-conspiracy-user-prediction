from training import grid_search

# TODO: Use argparse and logging libraries to make output cleaner.

test_params = {
    "sample_rate": 0.1,
    "model": "BERT_MLP",
    "split": [0.7, 0.15, 0.15],
    "epochs": 100,
    "max_features": 2**16,
    "sentences_length": 25,
    "transformer_model": "sentence-transformers/all-MiniLM-L6-v2",
}


def f_pth(f):
    """Quick helper function for readability"""
    return f"data/{f}.csv.gz"


BERT_DIR, POS_DIR, NEG_DIR = f_pth("bert"), f_pth("non-q-posts-v2"), f_pth("q-posts-v2")

grid_search_params = {
    "l2_penalty_weight": [0.001, 0.01, 0.1],
    "learning_rate": [0.0001, 0.001],
    "batch_size": [16, 32],
    "dropout_rate": [0.1, 0.25],
}

best_bert_mlp = grid_search.perform(
    static_params=test_params,
    grid_search_params=grid_search_params,
    pos_path=POS_DIR,
    neg_path=NEG_DIR,
    bert_path=BERT_DIR,
)
print(f"\n\n\n\n\nBEST BERT MLP:\n{best_bert_mlp}\n")
