import os
import torch
import lightning.pytorch as pl
import torchmetrics.functional as mF
from torch.optim import lr_scheduler
from torch import nn
from torch.nn import functional as F
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, random_split
from data.bids_dataset import BidsDataset
from models.unet_copilot import UNet

class LitUNet2DModule(pl.LightningModule):
    def __init__(self, in_channels, out_channels, start_lr=0.0001, lr_end_factor=1, n_classes:int=2, l2_reg_w=0.001):
        super(LitUNet2DModule, self).__init__()
        self.save_hyperparameters()

        self.model = UNet(in_channels, out_channels)
        self.start_lr = start_lr
        self.linear_end_factor = lr_end_factor
        self.l2_reg_w = l2_reg_w

        self.n_classes = n_classes

        self.train_step_outputs = {}
        self.val_step_outputs = {}

        self.softmax = nn.Softmax(dim=1)
        self.CEL = nn.CrossEntropyLoss()

    def on_fit_start(self):
        tb = self.logger.experiment  # noqa
        #
        layout_loss_train = [r"loss_train/dice_loss", "loss_train/l2_reg_loss", "loss_train/ce_loss"]
        layout_loss_val = [r"loss_val/dice_loss", "loss_val/l2_reg_loss", "loss_val/ce_loss"]
        layout_loss_merge = [r"loss/train_loss", "loss/val_loss"]
        layout_diceFG_merge = [r"diceFG/train_diceFG", "diceFG/val_diceFG"]
        layout_dice_merge = [r"dice/train_dice", "dice/val_dice"]
        layout_dice_p_cls_merge = [r"dice_p_cls/train_dice_p_cls", "dice_p_cls/val_dice_p_cls"]

        layout = {
            "loss_split": {
                "loss_train": ["Multiline", layout_loss_train],
                "loss_val": ["Multiline", layout_loss_val],
            },
            "loss_merge": {
                "loss": ["Multiline", layout_loss_merge],
            },
            "dice_merge": {
                "dice": ["Multiline", layout_dice_merge],
            },
            "diceFG_merge": {
                "diceFG": ["Multiline", layout_diceFG_merge],
            },
            "dice_p_cls_merge": {
                "dice_p_cls": ["Multiline", layout_dice_p_cls_merge],
            },
        }
        tb.add_custom_scalars(layout)        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        losses, logits, masks, pred_cls = self._shared_step(batch, batch_idx)
        loss = self._loss_merge(losses)
        metrics = self._shared_metric_step(loss, logits, masks, pred_cls)
        self.log('loss/train_loss', loss.detach().cpu(), batch_size=masks.shape[0], prog_bar=True)

        for k, v in losses.items():
            self.log(f"loss_train/{k}", v.detach().cpu(), batch_size=masks.shape[0], prog_bar=False)

        self._shared_metric_append(metrics, self.train_step_outputs)
        return loss

    def on_train_epoch_end(self) -> None:
        if len(self.train_step_outputs) > 0:
            metrics = self._shared_cat_metrics(self.train_step_outputs)

            self.log("dice/train_dice", metrics["dice"], on_epoch=True)
            self.log("diceFG/train_diceFG", metrics["diceFG"], on_epoch=True)
            self.logger.experiment.add_text("train_dice_p_cls", str(metrics["dice_p_cls"].tolist()), self.current_epoch)

            for i, score in enumerate(metrics["dice_p_cls"]):
                self.logger.experiment.add_scalar(f"dice_p_cls/train_dice_p_cls_{i}", score, self.current_epoch)
        self.train_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        losses, logits, masks, pred_cls = self._shared_step(batch, batch_idx)
        loss = self._loss_merge(losses)
        loss = loss.detach().cpu()
        metrics = self._shared_metric_step(loss, logits, masks, pred_cls)
        for l, v in losses.items():
            metrics[l] = v.detach().cpu()
        self._shared_metric_append(metrics, self.val_step_outputs)

    def on_validation_epoch_end(self) -> None:
        if len(self.val_step_outputs) > 0:
            metrics = self._shared_cat_metrics(self.val_step_outputs)

            for m, v in metrics.items():
                if "loss" in m:
                    if m == "loss":
                        self.log(f"loss/val_loss", v, on_epoch=True)
                    else:
                        self.log(f"loss_val/{m}", v, on_epoch=True)

            self.log("dice/val_dice", metrics["dice"], on_epoch=True)
            self.log("diceFG/val_diceFG", metrics["diceFG"], on_epoch=True)
            self.logger.experiment.add_text("val_dice_p_cls", str(metrics["dice_p_cls"].tolist()), self.current_epoch)

            for i, score in enumerate(metrics["dice_p_cls"]):
                self.logger.experiment.add_scalar(f"dice_p_cls/val_dice_p_cls_{i}", score, self.current_epoch)
        self.val_step_outputs.clear()

    def configure_optimizers(self): # next step to improve this
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)
        scheduler = lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1.0, end_factor=self.linear_end_factor, total_iters=20
        )
        if scheduler is not None:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}
        
    def loss(self, outputs, masks):
        ce_loss = self.CEL(outputs, masks)

        l2_reg = torch.tensor(0.0, device="cuda").to(non_blocking=True)
        for param in self.model.parameters():
            l2_reg += torch.norm(param).to(self.device, non_blocking=True)
        return {
            "ce_loss": ce_loss,
            "l2_reg_loss": (l2_reg * self.l2_reg_w),
        }
    
    def _loss_merge(self, losses: dict):
        return sum(losses.values())
    
    def _shared_step(self, batch, batch_idx, detach2cpu: bool =False):
        imgs, masks = batch         # unpacking the batch
        masks = masks.squeeze(1)    # removing the channel dimension
        logits = self.model(imgs)   # this implicitly calls the forward method

        losses = self.loss(logits, masks)

        with torch.no_grad():
            pred_x = self.softmax(logits)
            _, pred_cls = torch.max(pred_x, dim=1)
            del pred_x
            if detach2cpu:
                masks = masks.detach().cpu()
                logits = logits.detach().cpu()
                pred_cls = pred_cls.detach().cpu()
            dice = mF.dice(pred_cls, masks, num_classes=self.n_classes)
        return losses, logits, masks, pred_cls
    
    def _shared_metric_step(self, loss, logits, masks, pred_cls):
        dice = mF.dice(pred_cls, masks, num_classes=self.n_classes)
        diceFG = mF.dice(pred_cls, masks, num_classes=self.n_classes, ignore_index=0)   #ignore_index=0 means we ignore the background class
        dice_p_cls = mF.dice(pred_cls, masks, average=None, num_classes=self.n_classes) # average=None returns dice per class
        return {"loss": loss.detach().cpu(), "dice": dice, "diceFG": diceFG, "dice_p_cls": dice_p_cls}   

    def _shared_metric_append(self, metrics, outputs):
        for k, v in metrics.items():
            if k not in outputs:
                outputs[k] = []
            outputs[k].append(v)

    def _shared_cat_metrics(self, outputs):
        results = {}
        for m, v in outputs.items():
            stacked = torch.stack(v)
            results[m] = torch.mean(stacked) if m != "dice_p_cls" else torch.mean(stacked, dim=0)
        return results     

class BidsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str, contrast:str='t2f', format:str='fnio', do2D:bool=True, binary:bool=True, transform=None, padding=(256, 256), batch_size:int=2):
        super().__init__()
        self.data_dir = data_dir
        self.contrast = contrast
        self.format = format
        self.do2D = do2D
        self.binary = binary
        self.transform = transform
        self.padding = padding
        self.batch_size = batch_size

    def setup(self, stage: str = None) -> None:
        # Assign train/val datasets for use in dataloaders
        full_dataset = BidsDataset(self.data_dir, contrast=self.contrast, suffix=self.format, do2D=self.do2D, binary=self.binary, transform=self.transform,padding=self.padding)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(42)   # Set seed for reproducibility
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], generator)

    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=19)
    
    def val_dataloader(self) -> torch.Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=19)