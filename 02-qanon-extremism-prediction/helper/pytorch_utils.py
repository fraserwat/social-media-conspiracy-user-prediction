import torch
from helper.early_stopping import EarlyStopping


def accuracy(preds, y):
    print(preds.shape)
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float()
    return correct.sum() / len(correct)


def train_validate(
    model, training_loader, validation_loader, loss_function, optimizer, n_epochs=30
):
    # Initialise Early Stopping
    early_stopping = EarlyStopping(patience=3, verbose=True)

    for epoch in range(n_epochs):
        # Training Phase
        model.train()
        training_loss = 0.0

        for batch in training_loader:
            # Running through each batch in training loader, assuming response in last column.
            batch_inputs = batch[:, :-1]
            batch_targets = batch[:, -1]

            preds = model(batch_inputs)
            loss = loss_function(preds.squeeze(), batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

        # Validation Phase
        model.eval()
        validation_loss = 0.0

        with torch.no_grad():
            for batch in validation_loader:
                # Running through batches in validation set, assuming response in final col
                batch_inputs = batch[:, :-1]
                batch_targets = batch[:, -1]

                preds = model(batch_inputs)
                loss = loss_function(preds.squeeze(), batch_targets)
                validation_loss += loss.item()

        # Print stats
        tl, vl = round(training_loss / len(training_loader), 4), round(
            validation_loss / len(validation_loader), 4
        )
        print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {tl}, Valid Loss: {vl}")

        # Early Stopping
        early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        # Print loss every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {validation_loss:.4f}")


def test_model(model, testing_data_loader, loss_function):
    model.eval()
    test_loss = 0.0
    test_accuracy = 0.0

    with torch.no_grad():
        # Run through batches in the data loader
        for batch in testing_data_loader:
            # Assuming the response is in the last column.
            batch_inputs = batch[:, :-1]
            batch_targets = batch[:, -1]

            preds = model(batch_inputs)

            loss = loss_function(preds.squeeze(), batch_targets)
            acc = accuracy(preds.squeeze(), batch_targets)

            test_loss += loss.item()
            test_accuracy += acc.item()

    print(f"Test Loss: {test_loss/len(testing_data_loader):.4f}")
    print(f"Test Accuracy: {test_accuracy/len(testing_data_loader)*100:.2f}%")
