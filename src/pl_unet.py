import torch
import lightning.pytorch as pl
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from data.bids_dataset import BidsDataset
from models.unet_copilot import UNet

class LitUNet2DModule(pl.LightningModule):

    def __init__(self, in_channels, out_channels, learning_rate=0.001):
        super().__init__()
        self.model = UNet(in_channels, out_channels)
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, segs = batch
        segs = segs.squeeze(1)
        outputs = self.model(images)
        loss = self.criterion(outputs, segs)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, segs = batch
        segs = segs.squeeze(1)
        outputs = self.model(images)
        loss = self.criterion(outputs, segs)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class BidsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=2,  padding=(256, 256), contrast='t2f', binary=True):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.padding = padding
        self.contrast = contrast
        self.binary = binary

    def setup(self, stage: str = None) -> None:
        # Assign train/val datasets for use in dataloaders
        full_dataset = BidsDataset(self.data_dir, contrast=self.contrast, binary=self.binary, padding=self.padding)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], )

    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=19)
    
    def val_dataloader(self) -> torch.Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=19)
    
if __name__ == '__main__':
    # Set device automatically handled by PyTorch Lightning
    data_dir = "/media/DATA/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    model = LitUNet2DModule(in_channels=1, out_channels=2)
    data_module = BidsDataModule(data_dir, batch_size=2)

    # Trainer handles training loop
    trainer = pl.Trainer(max_epochs=5, default_root_dir='/media/DATA/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/', log_every_n_steps=1, accelerator='auto')
    trainer.fit(model, datamodule=data_module)