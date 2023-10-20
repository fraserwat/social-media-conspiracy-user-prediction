import torch
import torch.optim as optimizer
from models.RNN import RNN
from torch.utils.data import random_split, DataLoader, TensorDataset
from helper.pytorch_utils import train_validate, test_model


# Loading tensor input and response.
tensor = torch.load("data/BERT_RNN_X.pth")
response = torch.load("data/BERT_RNN_y.pth")

# Combine the tensor and response into a single dataset
combined_dataset = TensorDataset(tensor, response)

# Split data into train, validation, and test sets
train_size = int(0.7 * len(tensor))
valid_size = int(0.15 * len(tensor))
test_size = len(tensor) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(
    combined_dataset, [train_size, valid_size, test_size]
)

# Create Dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Now we can define the model
# Tensor shape is in format [batch_size, sequence_length, embedding_dim], so we want embedding_dim, tensor.shape[2]
model = RNN(input_size=tensor.shape[2])

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
