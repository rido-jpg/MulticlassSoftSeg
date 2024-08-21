import os
import sys
import torch
import torchmetrics
import numpy as np
import lightning.pytorch as pl
import torch.nn.functional as F
import torchmetrics.functional as mF
from pathlib import Path
from torch.optim import lr_scheduler
from torch import nn
from models.unet_copilot import UNet
from models.unet2D_H import Unet2D
from models.unet3D_H import Unet3D
#from medpy.metric.binary import assd
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, compute_average_surface_distance
#import medpy
#from panoptica.metrics import _compute_instance_average_symmetric_surface_distance

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from utils.brats_tools import soften_gt


class LitUNetModule(pl.LightningModule):
    def __init__(self, in_channels:int, out_channels:int, epochs:int, dim:int=32, groups:int=8, do2D:bool=False, binary:bool=False, soft:bool=False, one_hot:bool=True, sigma:float=0.1, start_lr:float=0.0001, lr_end_factor:float=0.01, n_classes:int=4, l2_reg_w:float=0.001, dsc_loss_w:float=1.0, ce_loss_w:float=1.0, soft_loss_w:float=0.0 ,  conf=None):
        super(LitUNetModule, self).__init__()

        self.save_hyperparameters()

        self.do2D = do2D
        self.binary = binary
        self.soft = soft
        self.one_hot = one_hot
        self.sigma = sigma

        self.start_lr = start_lr
        self.linear_end_factor = lr_end_factor
        self.l2_reg_w = l2_reg_w
        self.dsc_loss_w = dsc_loss_w
        self.ce_loss_w = ce_loss_w
        self.soft_loss_w = soft_loss_w

        self.epochs = epochs
        self.dim = dim
        self.groups = groups    # number of resnet block groups

        if self.do2D:
            #self.model = UNet(in_channels, out_channels)
            self.model = Unet2D(dim=self.dim, out_dim = out_channels, channels=in_channels)
        else:
            self.model = Unet3D(dim=self.dim, out_dim = out_channels, channels=in_channels, resnet_block_groups= self.groups)

        self.n_classes = n_classes
        self.transforms = None

        self.train_step_outputs = {}
        self.val_step_outputs = {}

        self.softmax = nn.Softmax(dim=1)
        self.CEL = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        #self.dice_loss = DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True, batch=True) # Monai Dice Loss
        #self.dice_loss = DiceLoss(to_onehot_y=True ,softmax=True) # Monai Dice Loss
        self.Dice = torchmetrics.Dice()
        self.DiceFG = torchmetrics.Dice(ignore_index=0) #ignore_index=0 means we ignore the background class
        #self.Dice_p_cls = torchmetrics.Dice(average=None, num_classes=self.n_classes)   # average='none' returns dice per class

    # # Operates on a mini-batch before it is transferred to the accelerator   
    # def on_before_batch_transfer(self, batch, dataloader_idx):
    #     if self.trainer.training:
    #         if self.transforms:
    #             batch = self.transforms(batch)
    #     return batch

    # # Operates on a mini-batch after it is transferred to the accelerator
    # def on_after_batch_transfer(self, batch, dataloader_idx):
    #     batch['x'] = gpu_transforms(batch['x'])
    #     return batch
    
    def on_fit_start(self):
        tb = self.logger.experiment  # noqa
        layout_loss_train = ["loss_train/dice_ET_loss", "loss_train/dice_WT_loss", "loss_train/dice_TC_loss", "loss_train/l2_reg_loss", "loss_train/ce_loss", "loss_train/mse_loss"]
        layout_loss_val = ["loss_val/dice_ET_loss", "loss_val/dice_WT_loss", "loss_val/dice_TC_loss", "loss_val/l2_reg_loss", "loss_val/ce_loss", "loss_val/mse_loss"]
        layout_loss_merge = ["loss/train_loss", "loss/val_loss"]
        layout_diceFG_merge = ["diceFG/train_diceFG", "diceFG/val_diceFG"]
        layout_dice_merge = ["dice/train_dice", "dice/val_dice"]
        layout_dice_ET_merge = ["dice_ET/train_dice_ET", "dice_ET/val_dice_ET"]
        layout_dice_TC_merge = ["dice_TC/train_dice_TC", "dice_TC/val_dice_TC"]
        layout_dice_WT_merge = ["dice_WT/train_dice_WT", "dice_WT/val_dice_WT"] 
        layout_dice_p_cls_merge = ["dice_p_cls/train_dice_p_cls", "dice_p_cls/val_dice_p_cls"]
        #layout_assd_merge = ["assd/train_assd", "assd/val_assd"]

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
            "Brats Dice Scores [ET=3, TC=1+3, WT=1+2+3]": {
                "dice_ET": ["Multiline", layout_dice_ET_merge],
                "dice_TC": ["Multiline", layout_dice_TC_merge],
                "dice_WT": ["Multiline", layout_dice_WT_merge],
            },
            # "assd_merge": {
            #     "assd": ["Multiline", layout_assd_merge],
            # },
        }
        tb.add_custom_scalars(layout)        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        losses, logits, masks, preds = self._shared_step(batch, batch_idx)
        loss = self._loss_merge(losses)
        metrics = self._shared_metric_step(loss, logits, masks, preds)
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
            #Brats Dice Scores
            self.log("dice_TC/train_dice_TC", metrics["dice_TC"], on_epoch=True)
            self.log("dice_ET/train_dice_ET", metrics["dice_ET"], on_epoch=True)
            self.log("dice_WT/train_dice_WT", metrics["dice_WT"], on_epoch=True)

            self.logger.experiment.add_text("train_dice_p_cls", str(metrics["dice_p_cls"].tolist()), self.current_epoch)

            for i, score in enumerate(metrics["dice_p_cls"]):
                self.logger.experiment.add_scalar(f"dice_p_cls/train_dice_p_cls_{i}", score, self.current_epoch)
            #self.log("assd/train_assd", metrics["assd"], on_epoch=True)
        self.train_step_outputs.clear()
    
    def validation_step(self, batch, batch_idx):
        losses, logits, masks, preds = self._shared_step(batch, batch_idx)
        loss = self._loss_merge(losses)
        loss = loss.detach().cpu()
        metrics = self._shared_metric_step(loss, logits, masks, preds)
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

            #Brats Dice Scores
            self.log("dice_TC/val_dice_TC", metrics["dice_TC"], on_epoch=True)
            self.log("dice_ET/val_dice_ET", metrics["dice_ET"], on_epoch=True)
            self.log("dice_WT/val_dice_WT", metrics["dice_WT"], on_epoch=True)

            self.logger.experiment.add_text("val_dice_p_cls", str(metrics["dice_p_cls"].tolist()), self.current_epoch)

            for i, score in enumerate(metrics["dice_p_cls"]):
                self.logger.experiment.add_scalar(f"dice_p_cls/val_dice_p_cls_{i}", score, self.current_epoch)
            #self.log("assd/val_assd", metrics["assd"], on_epoch=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self): # next step to improve this
        optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr, weight_decay=1e-5)
        scheduler = lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1.0, end_factor=self.linear_end_factor, total_iters=self.epochs
        )
        # scheduler = lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.epochs, eta_min=0
        # )
        if scheduler is not None:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}
        
    def loss(self, logits, preds, masks):
        # if self.one_hot:
        #     masks = torch.argmax(masks, dim=1).long()  # convert one-hot encoded masks back to integer masks
        # else:
        #     masks = masks.squeeze(1)    # remove the channel dimension for CrossEntropyLoss


        with torch.no_grad():   # is this torch.no_grad() necessary?
            masks = masks.squeeze(1)    # remove the channel dimension for CrossEntropyLoss
            oh_masks = F.one_hot(masks, num_classes=self.n_classes).permute(0, 4, 1, 2, 3).float()
            soft_masks = soften_gt(oh_masks.cpu(), self.sigma).to(self.device)
            del oh_masks
            
        # Regression Loss (SOFT LOSS)
        if self.soft_loss_w == 0:
            mse_loss = torch.tensor(0.0)
        else:
            mse_loss = self.MSE(logits, soft_masks) * self.soft_loss_w

        # Classification Losses (HARD LOSSES)
        if self.ce_loss_w == 0:
            ce_loss = torch.tensor(0.0)
        else:
            ce_loss = self.CEL(logits, masks) * self.ce_loss_w

        # Brats Dice Losses for subregions equally weighted
        if self.dsc_loss_w == 0:
            dice_ET_loss = dice_TC_loss = dice_WT_loss = torch.tensor(0.0)
        else:
            dice_ET_loss = (1 - self.Dice((preds == 3), (masks == 3))) * self.dsc_loss_w / 3
            dice_TC_loss = (1 - self.Dice((preds == 1) | (preds == 3), (masks == 1) | (masks == 3))) * self.dsc_loss_w / 3
            dice_WT_loss = (1 - self.Dice((preds > 0), (masks > 0))) * self.dsc_loss_w / 3

        # Weight Regularization
        l2_reg = torch.tensor(0.0, device=self.device).to(non_blocking=True)

        for param in self.model.parameters():
            l2_reg += torch.norm(param).to(self.device, non_blocking=True)

        return {
            "mse_loss": mse_loss,
            "ce_loss": ce_loss,
            "dice_ET_loss": dice_ET_loss,
            "dice_TC_loss": dice_TC_loss,
            "dice_WT_loss": dice_WT_loss,
            "l2_reg_loss": (l2_reg * self.l2_reg_w),
        }
    
    def _loss_merge(self, losses: dict):
        return sum(losses.values())
    
    def _shared_step(self, batch, batch_idx, detach2cpu: bool =False):
        imgs = batch['img']     # unpacking the batch
        masks = batch['seg']    # unpacking the batch
        logits = self.model(imgs)   # this implicitly calls the forward method
    
        del imgs # delete the images tensor to free up memory

        with torch.no_grad():

            if not self.binary:
                # create binary logits for the specific subregions ET, TC and WT by argmaxing the respective channels of the regions
                #logits_ET = torch.argmax(logits,)
                pass

            probs = self.softmax(logits)    # applying softmax to the logits to get probabilites
            preds = torch.argmax(probs, dim=1)   # getting the class with the highest probability
            del probs

            if detach2cpu:
                masks = masks.detach().cpu()
                logits = logits.detach().cpu()
                preds = preds.detach().cpu()

        losses = self.loss(logits,preds, masks)

        return losses, logits, masks, preds
    
    def _shared_metric_step(self, loss, logits, masks, preds):
        # if self.one_hot and not self.soft:
        #     masks = torch.argmax(masks, dim=1).long()  # convert one-hot encoded masks to integer masks

        # No squeezing of masks necessary as mF.dice implicitly squeezes dimension of size 1 (except batch size)
        # Overall Dice Scores
        dice = self.Dice(preds, masks)
        diceFG = self.DiceFG(preds, masks)
        dice_p_cls = mF.dice(preds, masks, average=None, num_classes=self.n_classes) # average=None returns dice per class

        # in the case that a class is not available in ground truth and preciction, the dice_p_cls would be NaN -> set it to 1, as it correctly predicts the absence
        for idx, score in enumerate(dice_p_cls):
            if score.isnan():
                dice_p_cls[idx] = 1.0

        # ET (Enhancing Tumor): label 3
        dice_ET = self.DiceFG((preds == 3), (masks == 3))

        # TC(Tumor Core): ET + NCR = label 1 + label 3
        dice_TC = self.DiceFG((preds == 1) | (preds == 3), (masks == 1) | (masks == 3))

        # WT (Whole Tumor): TC + ED = label 1 + label 2 + label 3
        dice_WT = self.DiceFG((preds > 0), (masks > 0))

        # print(f"dice: {dice}")
        # print(f"diceFG: {diceFG}")
        # print(f"dice_p_cls: {dice_p_cls}")
        # print(f"_shared_metric_step():")
        # print(f"preds shape: {preds.shape}, masks shape: {masks.shape}")
        # # print the unique values in the masks and preds tensors
        # print(f"unique values in masks: {np.unique(masks.cpu().numpy())}")
        # print(f"unique values in preds: {np.unique(preds.cpu().numpy())}")
        # with torch.no_grad():
        #     # Convert to one-hot encoded tensors
        #     masks_one_hot = self._to_one_hot(masks, self.n_classes)
        #     preds_one_hot = self._to_one_hot(preds, self.n_classes)
        #     # print shapes of one-hot encoded masks and preds tensors
        #     # print(f"masks_one_hot shape: {masks_one_hot.shape}, preds_one_hot shape: {preds_one_hot.shape}")
        #     # # print the unique values in the one-hot encoded masks and preds tensors
        #     # print(f"unique values in masks_one_hot: {np.unique(masks_one_hot.cpu().numpy())}")
        #     # print(f"unique values in preds_one_hot: {np.unique(preds_one_hot.cpu().numpy())}")
        #     # Filter out empty classes
        #     preds_one_hot, masks_one_hot = self._filter_empty_classes(preds_one_hot, masks_one_hot)
        # # calculate average symmetric surface distance (assd)
        # assd = compute_average_surface_distance(preds_one_hot, masks_one_hot, symmetric=True)
        # print(f"assd: {assd}")
        # del masks_one_hot, preds_one_hot
        return {
            "loss": loss.detach().cpu(), 
            "dice": dice.detach().cpu(), 
            "diceFG": diceFG.detach().cpu(), 
            "dice_p_cls": dice_p_cls.detach().cpu(), 
            "dice_ET": dice_ET.detach().cpu(),
            "dice_TC": dice_TC.detach().cpu(),
            "dice_WT": dice_WT.detach().cpu(),
            #"assd": assd
        }           

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
    
    def _filter_empty_classes(self, preds_one_hot, masks_one_hot):
        """
        Filter out classes that are completely absent in either the predictions or ground truth.
        
        Args:
            preds_one_hot: One-hot encoded predictions of shape (batch_size, num_classes, height, width).
            masks_one_hot: One-hot encoded ground truth of shape (batch_size, num_classes, height, width).
        
        Returns:
            Filtered predictions and ground truth with non-empty classes.
        """
        non_empty_classes = (masks_one_hot.sum(dim=[0, 2, 3]) != 0) & (preds_one_hot.sum(dim=[0, 2, 3]) != 0)
        preds_one_hot = preds_one_hot[:, non_empty_classes]
        masks_one_hot = masks_one_hot[:, non_empty_classes]
        return preds_one_hot, masks_one_hot