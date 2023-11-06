import torch
from processing import load, split, tensor


def input_fn(pos_path, neg_path, bert_path, params):
    # source data provided is different for bert / non-bert
    print("Loading QAnon dataset and creating df...")
    if params.get("model", "task_specific").lower().startswith("bert"):
        print("Making bert dataset")
        df = load.load_bert_data_to_df(bert_path, params)
    else:
        print("Making word embedding dataset")
        df = load.load_all_data_task_specific(pos_path, neg_path, params)

    train, test = split.train_val_test_split(df, params=params)

    if params.get("model", "MLP").upper() == "BERT_LSTM":
        # Actual BERT embeddings not done at this stage!! Purely data prep + tensor construction.
        # TODO: Add BERT embeddings for LSTM.
        words_train = tensor.convert_bert_lstm_to_tensor(train, params)
        words_test = tensor.convert_bert_lstm_to_tensor(test, params)
    elif params.get("model", "MLP").upper() in ["BERT_RNN", "BERT_MLP"]:
        words_train = tensor.convert_bert_rnn_mlp_to_tensor(train, params)
        words_test = tensor.convert_bert_rnn_mlp_to_tensor(test, params)
    else:
        # Directly convert 'words' column to a list as PyTorch doesn't support native string tensor
        words_train = train["words"].tolist()
        words_test = test["words"].tolist()

    labels_train = torch.tensor(train["q_level"].values, dtype=torch.float32)
    labels_test = torch.tensor(test["q_level"].values, dtype=torch.float32)

    inputs = {
        "train": [words_train, labels_train],
        "test": [words_test, labels_test],
    }

    print("Done data processing")

    return inputs
