import torch
from torch.utils.data import DataLoader  # , Dataset
from training.train_test import train_validate, test_model


def sentence_embedded_model(input_data, transformer_model, model, params):
    # custom collation for the dataloader to stack raw text data (create_data_loader)
    def collate_fn(batch):
        texts, labels = zip(*batch)
        return list(texts), torch.tensor(labels)

    # Function to create DataLoader from given texts and labels.
    def create_data_loader(texts, labels, batch_size):
        # Pair each text with its corresponding label
        dataset = list(zip(texts, labels))
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
        )
        return loader

    # Extract texts and labels for training, validation, and testing
    train_texts, train_labels = input_data["train"]
    val_texts, val_labels = input_data["val"]
    test_texts, test_labels = input_data["test"]

    # Create DataLoaders for train, validation, and test sets
    batch_size = params.get("batch_size", 32)
    train_loader = create_data_loader(train_texts, train_labels, batch_size)
    val_loader = create_data_loader(val_texts, val_labels, batch_size)
    test_loader = create_data_loader(test_texts, test_labels, batch_size)

    # Initialise model
    model = model(
        bert_model_path=transformer_model,
        dropout_rate=params.get("dropout_rate", 0.0),
    )

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
