import torch
import torch.optim as optimizer
from models.MLP import MLP
from helper.pytorch_utils import train_validate, test_model
from torch.utils.data import random_split, DataLoader


# Splitting tensor into input and target (assuming the last column is the target)
tensor = torch.load("data/MLP.pth")
inputs = tensor[:, :-1]  # Feature columns
targets = tensor[:, -1].unsqueeze(1)  # Target column, ensure it's a column vector

# Split data into train, validation, and test sets
train_size = int(0.7 * len(tensor))
valid_size = int(0.15 * len(tensor))
test_size = len(tensor) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(
    tensor, [train_size, valid_size, test_size]
)

# Create Dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Now we can define the model
model = MLP(input_size=inputs.shape[1])

# L2 Regularisation is given through weight decay.
adam_optimizer = optimizer.Adam(model.parameters(), weight_decay=1e-4)


# Training Loop
num_epochs = 30  # Adjust based on your needs
log_loss = torch.nn.BCELoss()

# Run the training and validation loop.
train_validate(
    model=model,
    training_loader=train_loader,
    validation_loader=valid_loader,
    loss_function=log_loss,
    optimizer=adam_optimizer,
)

# Load the last checkpoint with the best model.
model.load_state_dict(torch.load("checkpoint.pth"))

# Finally, evaluate the model.
test_model(model=model, testing_data_loader=test_loader, loss_function=log_loss)
