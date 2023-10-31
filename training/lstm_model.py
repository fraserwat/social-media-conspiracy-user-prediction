import torch

# from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from models.LSTM import BaselineLSTM
from training.pytorch_utils import train_validate, test_model


def word_lstm_model(input_data, vectorize_layer, params):
    # Combine all datasets into a single generator
    all_texts = (
        text for dataset in ["train", "val", "test"] for text in input_data[dataset][0]
    )

    # Calculate the max length considering the upper limit
    max_length = min(
        max(len(vectorize_layer(text)) for text in all_texts),
        params.get("max_features", float("inf")),
    )

    def create_data_loader(data, sequence_length):
        data_tokenized = [vectorize_layer(text) for text in data[0]]
        data_padded = torch.zeros(len(data_tokenized), sequence_length)

        for i, tokenized_text in enumerate(data_tokenized):
            length = min(len(tokenized_text), sequence_length)
            data_padded[i, :length] = torch.tensor(
                tokenized_text[:length], dtype=torch.float
            )

        dataset = TensorDataset(data_padded, torch.tensor(data[1], dtype=torch.float))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        return loader

    train_loader = create_data_loader(input_data["train"], max_length)
    val_loader = create_data_loader(input_data["val"], max_length)
    test_loader = create_data_loader(input_data["test"], max_length)

    # Initialise MLP model
    model = BaselineLSTM(input_size=max_length)

    # Use Binary Cross Entropy Loss and Adam optimizer
    loss = torch.nn.BCELoss()
    adam = torch.optim.Adam(
        model.parameters(), weight_decay=params.get("l2_reg_lambda", 0)
    )

    # Train and validate the model
    train_validate(model, train_loader, val_loader, loss, adam)

    # Test the model
    test_model(model, test_loader, loss)
