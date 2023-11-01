import torch
from torch.utils.data import DataLoader, TensorDataset
from training.pytorch_utils import train_validate, test_model


def word_embedded_model(input_data, vectorize_layer, model, params):
    # Combine all datasets into a single generator
    # TODO: Check same process used in paper. Getting really volitile results from word LSTM
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

        dataset = TensorDataset(data_padded, data[1].clone().detach())
        loader = DataLoader(
            dataset, batch_size=params.get("batch_size", 32), shuffle=True
        )

        return loader

    train_loader = create_data_loader(input_data["train"], max_length)
    val_loader = create_data_loader(input_data["val"], max_length)
    test_loader = create_data_loader(input_data["test"], max_length)

    # Initialise MLP model
    model = model(input_size=max_length, dropout_rate=params.get("dropout_rate", 0.0))

    # Use Binary Cross Entropy Loss and Adam optimizer
    loss = torch.nn.BCELoss()
    adam = torch.optim.Adam(
        model.parameters(),
        weight_decay=params.get("l2_penalty_weight", 0),
        lr=params.get("learning_rate", 0.001),
    )

    # Train and validate the model
    checkpoint_path = train_validate(model, train_loader, val_loader, loss, adam)

    # Test the model
    loss, accuracy, recall, f1 = test_model(
        model, test_loader, loss, checkpoint_path=checkpoint_path
    )

    return loss, accuracy, recall, f1
