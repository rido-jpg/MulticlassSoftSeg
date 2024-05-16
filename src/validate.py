import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.bids_dataset import BidsDataset
from models.unet_copilot import UNet

def dice_coefficient(predictions, masks, smooth=1):
    intersection = (predictions * masks).sum()
    return (2. * intersection + smooth) / (predictions.sum() + masks.sum() + smooth)

def validate(root_dir, batch_size, model_path):
    logging.basicConfig(level=logging.INFO)
    
    # Initialize dataset and dataloader
    validation_dataset = BidsDataset(root_dir=root_dir, resize=(256, 256))
    val_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model and load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=1, out_channels=2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Validation loop
    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks.squeeze(1))
            val_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            dice_score += dice_coefficient(predictions, masks)

    avg_loss = val_loss / len(val_loader)
    avg_dice = dice_score / len(val_loader)
    logging.info(f'Validation Loss: {avg_loss}, Dice Score: {avg_dice}')
