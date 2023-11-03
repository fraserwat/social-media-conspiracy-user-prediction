import torch
from sklearn.metrics import recall_score, f1_score
from training.early_stopping import EarlyStopping


def accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)


def train_validate(
    model,
    training_loader,
    validation_loader,
    loss_function,
    optimizer,
    n_epochs=30,
):
    # Initialize Early Stopping
    early_stopping = EarlyStopping(patience=5, verbose=False)
    checkpoint_path = None

    for epoch in range(n_epochs):
        # Training Phase
        model.train()
        training_loss = 0.0

        for batch_inputs, batch_targets in training_loader:
            preds = model(batch_inputs)
            loss = loss_function(preds.squeeze(), batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # Validation Phase
        model.eval()
        validation_loss = 0.0

        # print("validation process beginning...")

        with torch.no_grad():
            for batch_inputs, batch_targets in validation_loader:
                preds = model(batch_inputs)
                loss = loss_function(preds.squeeze(), batch_targets)
                validation_loss += loss.item()

        # Print stats
        tl, vl = round(training_loss / len(training_loader), 4), round(
            validation_loss / len(validation_loader), 4
        )
        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {tl}, Valid Loss: {vl}")

        # Early Stopping
        checkpoint_path = early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return checkpoint_path


def test_model(model, testing_data_loader, loss_function, checkpoint_path=None):
    if checkpoint_path is not None:
        print(f"Loading checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        # Run through batches in the data loader
        for batch_inputs, batch_targets in testing_data_loader:
            preds = model(batch_inputs)
            rounded_preds = torch.round(preds.squeeze())
            loss = loss_function(preds.squeeze(), batch_targets)
            acc = accuracy(preds.squeeze(), batch_targets)

            test_loss += loss.item()
            test_accuracy += acc.item()

            # Append rounded predictions and true labels for this batch (for recall / f1)
            all_preds.extend(rounded_preds.cpu().numpy())
            all_labels.extend(batch_targets.cpu().numpy())

    # Calculate Recall and F1 score
    test_recall = recall_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds)

    final_loss = test_loss / len(testing_data_loader)
    final_accuracy = test_accuracy / len(testing_data_loader) * 100

    # print(f"Test Loss: {final_loss:.4f}")
    print(f"Test Accuracy: {final_accuracy:.2f}")
    print(f"Test F1 Score: {test_f1:.2f}")

    return final_loss, final_accuracy, test_recall, test_f1
