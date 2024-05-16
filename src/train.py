import torch
from torch.utils.data import DataLoader
from data.bids_dataset import BidsDataset
from models.unet_copilot import UNet
import datetime

import torch.nn as nn
import torch.optim as optim

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set hyperparameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Create dataset and dataloader
dataset_root = "/media/DATA/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
train_dataset = BidsDataset(dataset_root, contrast='t2f', do2D=True, binary=True, padding=(256, 256))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

model = UNet(in_channels=1, out_channels=2).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Initialize variables
epoch_loss = 0.0
total_pixels = 0
correct_pixels = 0
train_losses = []
train_accs = []
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    total_pixels = 0
    correct_pixels = 0
    for images, labels in train_loader:
        # Move images and labels to device
        images = images.to(device)
        labels = labels.to(device)
        labels = labels.squeeze(1)  # Squeeze the channel dimension of labels

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
        total_pixels += images.numel()                          # Number of pixels in the batch 
        correct_pixels += (predicted == labels).sum().item()    # Number of correct pixels in the batch


    # Calculate epoch loss and accuracy
    epoch_loss /= (len(train_loader) * batch_size)          # Average loss over all batches in one epoch
    train_acc = (correct_pixels / total_pixels)             # Accuracy over all pixels in one epoch


    # Append epoch loss and accuracy to lists
    train_losses.append(epoch_loss)                 # Not sure if this is correct
    train_accs.append(train_acc)                    # Not sure if this is correct

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