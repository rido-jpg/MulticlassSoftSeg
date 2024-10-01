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
from argparse import Namespace
from models.unet_copilot import UNet
from models.unet2D_H import Unet2D
from models.unet3D_H import Unet3D
#from medpy.metric.binary import assd
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, compute_average_surface_distance
from utils.losses import AdapWingLoss   # complex region selective loss
from utils.adaptive_wing_loss import AdaptiveWingLoss # standard implementation
from utils.dice import MemoryEfficientSoftDiceLoss
from TPTBox import np_utils
from panoptica.metrics import _compute_dice_coefficient
#import medpy
#from panoptica.metrics import _compute_instance_average_symmetric_surface_distance

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from utils.brats_tools import soften_gt


class LitUNetCityModule(pl.LightningModule):
    def __init__(self, opt: Namespace, in_channels: int, out_channels: int, binary: bool=False, n_classes: int=4):
        super(LitUNetCityModule, self).__init__()

        self.save_hyperparameters()

        self.opt = opt

        self.do2D = opt.do2D
        self.binary = binary
        self.soft = opt.soft
        self.one_hot = opt.one_hot
        self.sigma = opt.sigma
        self.dilate = opt.dilate
        self.final_activation = opt.activation

        self.start_lr = opt.lr
        self.linear_end_factor = opt.lr_end_factor
        self.l2_reg_w = opt.l2_reg_w
        self.dsc_loss_w = opt.dsc_loss_w
        self.ce_loss_w = opt.ce_loss_w
        self.soft_loss_w = opt.soft_loss_w
        self.hard_loss_w = opt.hard_loss_w
        self.mse_loss_w = opt.mse_loss_w
        self.adw_loss_w = opt.adw_loss_w
        self.soft_dice_loss_w = opt.soft_dice_loss_w

        self.epochs = opt.epochs
        self.dim = opt.dim
        self.groups = opt.groups    # number of resnet block groups

        if self.do2D:
            #self.model = UNet(in_channels, out_channels)
            self.model = Unet2D(dim=self.dim, out_dim = out_channels, channels=in_channels, resnet_block_groups= self.groups)
        else:
            self.model = Unet3D(dim=self.dim, out_dim = out_channels, channels=in_channels, resnet_block_groups= self.groups)

        self.n_classes = n_classes
        self.transforms = None

        self.train_step_outputs = {}
        self.val_step_outputs = {}

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()
        self.BCEL = nn.BCEWithLogitsLoss()
        self.MSE = nn.MSELoss()
        self.SoftDice = MemoryEfficientSoftDiceLoss(apply_nonlin= self.softmax,batch_dice=False, do_bg=True, smooth=1e-5, ddp=False)
        #self.ADWL = AdapWingLoss(epsilon=1, alpha=2.1, theta=0.5, omega=8)     # complex region selective implementation
        self.ADWL = AdaptiveWingLoss(epsilon=1, alpha=2.1, theta=0.5, omega=8)  # standard implementation
        #self.dice_loss = DiceLoss(smooth_nr=0, smooth_dr=1e-5, to_onehot_y=False, sigmoid=True, batch=True) # Monai Dice Loss
        self.DiceLoss = DiceLoss(sigmoid = True) # Monai Dice Loss on single channel -> to onehot = True and include background = False not neccesary and wouldn't work
        self.Dice = torchmetrics.Dice()
        self.DiceFG = torchmetrics.Dice(ignore_index=0)

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
        #
        layout_loss_train = ["loss_train/dice_loss", "loss_train/l2_reg_loss", "loss_train/ce_loss", "loss_train/mse_loss", "loss_train/adw_loss"]
        layout_loss_val = ["loss_val/dice_loss", "loss_val/l2_reg_loss", "loss_val/ce_loss", "loss_val/mse_loss", "loss_val/adw_loss"]
        layout_loss_merge = ["loss/train_loss", "loss/val_loss"]
        layout_diceFG_merge = ["diceFG/train_diceFG", "diceFG/val_diceFG"]
        layout_dice_merge = ["dice/train_dice", "dice/val_dice"]
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
            # "assd_merge": {
            #     "assd": ["Multiline", layout_assd_merge],
            # },
        }
        tb.add_custom_scalars(layout)        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        losses, logits, masks, preds = self._shared_step(batch, detach2cpu=True)
        loss = self._loss_merge(losses)
        metrics = self._shared_metric_step(loss, logits, masks, preds)
        self.log('loss/train_loss', loss.detach(), batch_size=masks.shape[0], prog_bar=True)

        for k, v in losses.items():
            self.log(f"loss_train/{k}", v.detach(), batch_size=masks.shape[0], prog_bar=False)

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
            #self.log("assd/train_assd", metrics["assd"], on_epoch=True)
        self.train_step_outputs.clear()
    
    def validation_step(self, batch):
        losses, logits, masks, preds = self._shared_step(batch, detach2cpu=True)
        loss = self._loss_merge(losses)
        loss = loss.detach()
        metrics = self._shared_metric_step(loss, logits, masks, preds)
        for l, v in losses.items():
            metrics[l] = v.detach()
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
            #self.log("assd/val_assd", metrics["assd"], on_epoch=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr, weight_decay=1e-5)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.start_lr, weight_decay=self.l2_reg_w)
        scheduler = lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1.0, end_factor=self.linear_end_factor, total_iters=self.epochs
        )
        # scheduler = lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.epochs, eta_min=0
        # )
        # Explicitly specify that the scheduler should update after the optimizer step
        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "reduce_on_plateau": False,  # Not using ReduceLROnPlateau
                },
            }
            
        return {"optimizer": optimizer}
        
    def loss(self, logits, masks, soft_masks):
                    
        # Regression Loss (SOFT LOSS)
        if self.soft_loss_w == 0 or self.mse_loss_w == 0:
            mse_loss = torch.tensor(0.0)
        else:
            mse_loss = self.MSE(logits, soft_masks) * self.mse_loss_w * self.soft_loss_w

        if self.adw_loss_w == 0 or self.soft_loss_w == 0:
            adw_loss = torch.tensor(0.0)
        else:
            adw_loss = self.ADWL(logits, soft_masks) * self.adw_loss_w * self.soft_loss_w

        # Classification Losses (HARD LOSSES)
        if self.ce_loss_w == 0 or self.hard_loss_w == 0:
            ce_loss = torch.tensor(0.0)
        else:
            float_masks = masks.to(torch.float16)
            ce_loss = self.BCEL(logits, float_masks) * self.ce_loss_w * self.hard_loss_w

        # Brats Dice Losses for subregions equally weighted
        if self.dsc_loss_w == 0 or self.hard_loss_w == 0:
            dice_loss = torch.tensor(0)
        else:
            dice_loss = (self.DiceLoss(logits, masks)) * self.dsc_loss_w * self.hard_loss_w

        # Brats Soft Dice Losses for subregions equally weighted
        if self.soft_dice_loss_w == 0:
            soft_dice_loss = torch.tensor(0.0)
        else:
            soft_dice_loss = self.SoftDice(logits, masks) * self.soft_dice_loss_w

        # # Weight Regularization
        # l2_reg = torch.tensor(0.0, device=self.device).to(non_blocking=True)

        # for param in self.model.parameters():
        #     l2_reg += torch.norm(param).to(self.device, non_blocking=True)

        l2_reg = torch.tensor(0.0)    

        return {
            "adw_loss": adw_loss,
            "mse_loss": mse_loss,
            "ce_loss": ce_loss,
            "soft_dice_loss": soft_dice_loss,
            "dice_loss": dice_loss,
            "l2_reg_loss": (l2_reg * self.l2_reg_w),
        }
    
    def _loss_merge(self, losses: dict):
        return sum(losses.values())
    
    def _shared_step(self, batch, detach2cpu: bool =False):
        imgs = batch['img']     # unpacking the batch
        masks = batch['seg']    # unpacking the batch
        soft_masks = batch['soft_seg']  # unpacking the batch
        logits = self.model(imgs)   # this implicitly calls the forward method

        losses = self.loss(logits, masks, soft_masks)
    
        del imgs # delete the images tensor to free up memory
        with torch.no_grad():
            if not self.binary:
                # create binary logits for the specific subregions ET, TC and WT by argmaxing the respective channels of the regions
                #logits_ET = torch.argmax(logits,)
                pass
            if self.final_activation == 'sigmoid':
                probs = self.sigmoid(logits)

            elif self.final_activation == "softmax":
                probs = self.softmax(logits)    # applying softmax to the logits to get probabilites

            elif self.final_activation == "relu":
                if bool(self.relu(logits).max()): # checking if the max value of the relu is not zero
                    probs = self.relu(logits)/self.relu(logits).max()
                else: 
                    probs = self.relu(logits)

            preds = torch.argmax(probs, dim=1)   # getting the class with the highest probability

            del probs

            if detach2cpu:
                masks = masks.detach()
                soft_masks = soft_masks.detach()
                logits = logits.detach()
                preds = preds.detach()

        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)

        for k, v in losses.items():
            if v is not isinstance(v, bool):
                if v.requires_grad == False and v != 0:
                    print(f"{k} = {v} requires grad:")
                    print(f"{v.requires_grad}")

        return losses, logits, masks, preds
    
    def _shared_metric_step(self, loss, logits, masks, preds):
        # if self.one_hot and not self.soft:
        #     masks = torch.argmax(masks, dim=1).long()  # convert one-hot encoded masks to integer masks

        # No squeezing of masks necessary as mF.dice implicitly squeezes dimension of size 1 (except batch size)
        # Overall Dice Scores
        dice = self.Dice(preds, masks)
        diceFG = self.DiceFG(preds, masks)
        if self.n_classes == 1:
            dice_p_cls = mF.dice(preds, masks, average=None, num_classes=self.n_classes + 1) # average=None returns dice per class
        else:
            dice_p_cls = mF.dice(preds, masks, average=None, num_classes=self.n_classes) # average=None returns dice per class

        # in the case that a class is not available in ground truth and prediction, the dice_p_cls would be NaN -> set it to 1, as it correctly predicts the absence
        for idx, score in enumerate(dice_p_cls):
            if score.isnan():
                dice_p_cls[idx] = 1.0

        return {    # does it make sense detaching all these scores and losses?
            "loss": loss.detach(), 
            "dice": dice.detach(),
            "diceFG": diceFG.detach(),
            "dice_p_cls": dice_p_cls.detach(),
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
    
    def _prepare_region_logits(self, logits, target_classes):
        """
        Efficiently sum the logits for the given target classes.
        """
        return logits[:, target_classes].sum(dim=1, keepdim=True)