import torch
from torch.utils.data import DataLoader
from data.make_dataset import BidsDataset
from models.unet_copilot import UNet
import datetime

import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
batch_size = 4
learning_rate = 0.001
num_epochs = 10

# Create dataset and dataloader
dataset = BidsDataset()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = UNet().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize variables
epoch_loss = 0.0
total = 0
correct = 0
train_losses = []
train_accs = []
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update epoch loss
        epoch_loss += loss.item()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate epoch loss and accuracy
    epoch_loss /= len(dataloader)
    train_acc = correct / total

    # Append epoch loss and accuracy to lists
    train_losses.append(epoch_loss)
    train_accs.append(train_acc)

    # Print epoch loss and accuracy
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Accuracy: {train_acc}")

# Save the trained model
torch.save(model.state_dict(), "trained_model.pth")

# Log training progress and model performance
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_file = "/media/DATA/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/training_log.txt"

with open(log_file, "a") as f:
    f.write(f"Training log - {timestamp}\n")
    f.write(f"Num epochs: {num_epochs}\n")
    f.write(f"Batch size: {batch_size}\n")
    f.write(f"Learning rate: {learning_rate}\n")
    f.write(f"Final loss: {train_losses[-1]}\n")
    f.write(f"Final accuracy: {train_accs[-1]}\n")
    f.write("\n")