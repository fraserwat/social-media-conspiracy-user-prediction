import torch
from processing import download, load, tensor, split
from embedding import vectorise, trans
from training import word_embedding
from models import MLP, RNN, LSTM

# TODO: Maybe use argparse to make this cleaner.
# TODO: What is the difference between the `logging` library and normal print statements

test_params = {
    "sample_rate": 0.15,
    # "embedding": None,
    "model": "MLP",
    "split": [0.7, 0.15, 0.15],
    "epochs": 100,
    "max_features": 2**16,
}


def f_pth(f):
    """Quick helper function for readability"""
    return f"data/{f}.csv.gz"


BERT_DIR, POS_DIR, NEG_DIR = f_pth("bert"), f_pth("non-q-posts-v2"), f_pth("q-posts-v2")


def input_fn(pos_path, neg_path, bert_path, params):
    # source data provided is different for bert / non-bert

    print("Loading QAnon dataset and creating df...")
    if params.get("model", "task_specific").lower().startswith("bert"):
        print("Making bert dataset")
        df = load.load_bert_data_to_df(bert_path, params)
    else:
        print("Making word embedding dataset")
        df = load.load_all_data_task_specific(pos_path, neg_path, params)

    train, val, test = split.train_val_test_split(df, params=params)

    if params.get("model", "MLP").upper() == "BERT_LSTM":
        # Actual BERT embeddings not done at this stage!! Purely data prep + tensor construction.
        words_train = tensor.convert_bert_lstm_to_tensor(train, params)
        words_val = tensor.convert_bert_lstm_to_tensor(val, params)
        words_test = tensor.convert_bert_lstm_to_tensor(test, params)
    elif params.get("model", "MLP").upper() in ["BERT_RNN", "BERT_MLP"]:
        words_train = tensor.convert_bert_rnn_mlp_to_tensor(train, params)
        words_val = tensor.convert_bert_rnn_mlp_to_tensor(val, params)
        words_test = tensor.convert_bert_rnn_mlp_to_tensor(test, params)
    else:
        # Directly convert 'words' column to a list as PyTorch doesn't support native string tensor
        words_train = train["words"].tolist()
        words_val = val["words"].tolist()
        words_test = test["words"].tolist()

    labels_train = torch.tensor(train["q_level"].values, dtype=torch.float32)
    labels_val = torch.tensor(val["q_level"].values, dtype=torch.float32)
    labels_test = torch.tensor(test["q_level"].values, dtype=torch.float32)

    inputs = {
        "train": [words_train, labels_train],
        "val": [words_val, labels_val],
        "test": [words_test, labels_test],
    }

    print("Done data processing")

    return inputs


def model_fn(inputs: dict, params: dict):
    model_type = params.get("model", "").upper()
    max_features = params.get("max_features", 10**4)

    if "BERT" in model_type:
        # Sentence BERT embeddings
        if model_type.endswith("MLP"):
            pass
        elif model_type.endswith("RNN"):
            pass
        elif model_type.endswith("LSTM"):
            pass
        else:
            raise NotImplementedError
    else:
        # Word embeddings - as we are embedding manually, need to create vectorised layer
        vectorise_fn = vectorise.create_vectorised_layer(
            # inputs["train"][0] will be the first item in the dict, words_train
            words=inputs["train"][0],
            max_features=max_features,
        )

        if model_type.endswith("MLP"):
            word_embedding.word_embedded_model(
                input_data=inputs,
                vectorize_layer=vectorise_fn,
                model=MLP.MLP,
                params=test_params,
            )
        elif model_type.endswith("RNN"):
            word_embedding.word_embedded_model(
                input_data=inputs,
                vectorize_layer=vectorise_fn,
                model=RNN.RNN,
                params=test_params,
            )
        elif model_type.endswith("LSTM"):
            word_embedding.word_embedded_model(
                input_data=inputs,
                vectorize_layer=vectorise_fn,
                model=LSTM.BaselineLSTM,
                params=test_params,
            )
        else:
            raise NotImplementedError

    return None


# download.download()

test_params["model"] = "MLP"
tensor_dict = input_fn(
    pos_path=POS_DIR, neg_path=NEG_DIR, bert_path=BERT_DIR, params=test_params
)
model_fn(tensor_dict, test_params)

# test_params["model"] = "RNN"
# tensor_dict = input_fn(
#     pos_path=POS_DIR, neg_path=NEG_DIR, bert_path=BERT_DIR, params=test_params
# )
# model_fn(tensor_dict, test_params)

# test_params["model"] = "LSTM"
# tensor_dict = input_fn(
#     pos_path=POS_DIR, neg_path=NEG_DIR, bert_path=BERT_DIR, params=test_params
# )
# model_fn(tensor_dict, test_params)
