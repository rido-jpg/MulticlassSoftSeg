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



class LitUNetModule(pl.LightningModule):
    def __init__(self, opt: Namespace, in_channels: int, out_channels: int):
        super(LitUNetModule, self).__init__()

        self.save_hyperparameters()

        self.opt = opt

        self.do2D = opt.do2D
        self.binary = opt.binary
        self.soft = opt.soft
        self.one_hot = opt.one_hot
        self.sigma = opt.sigma
        self.dilate = opt.dilate
        self.final_activation = opt.activation

        self.out_channels = out_channels

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
            h, w = self.opt.resize
            self.tensor_shape = [self.opt.bs, 2, h, w]

            self.img_area = h*w
            self.example_input_array = torch.rand([self.opt.bs, in_channels, h, w])

            self.model = Unet2D(dim=self.dim, out_dim = self.out_channels, channels=in_channels, resnet_block_groups= self.groups)
        else:
            h, w, d = self.opt.resize
            self.tensor_shape = [self.opt.bs, 2, h, w, d]

            self.img_area = h*w*d
            self.example_input_array = torch.rand([self.opt.bs, in_channels, h, w, d])

            self.model = Unet3D(dim=self.dim, out_dim = self.out_channels, channels=in_channels, resnet_block_groups= self.groups)

        self.transforms = None

        self.train_step_outputs = {}
        self.val_step_outputs = {}

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.CEL = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.SoftDice = MemoryEfficientSoftDiceLoss(apply_nonlin= self.softmax,batch_dice=False, do_bg=True, smooth=1e-5, ddp=False)
        #self.ADWL = AdapWingLoss(epsilon=1, alpha=2.1, theta=0.5, omega=8)     # complex region selective implementation
        self.ADWL = AdaptiveWingLoss(epsilon=1, alpha=2.1, theta=0.5, omega=8)  # standard implementation
        self.MonaiDiceBratsLoss = DiceLoss(softmax=True, to_onehot_y=True) 
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
        layout_loss_train = ["loss_train/dice_ET_loss", "loss_train/dice_WT_loss", "loss_train/dice_TC_loss", "loss_train/l2_reg_loss", "loss_train/ce_loss", "loss_train/mse_loss", "loss_train/adw_loss"]
        layout_loss_val = ["loss_val/dice_ET_loss", "loss_val/dice_WT_loss", "loss_val/dice_TC_loss", "loss_val/l2_reg_loss", "loss_val/ce_loss", "loss_val/mse_loss", "loss_val/adw_loss"]
        layout_loss_merge = ["loss/train_loss", "loss/val_loss"]
        layout_diceFG_merge = ["diceFG/train_diceFG", "diceFG/val_diceFG"]
        layout_dice_merge = ["dice/train_dice", "dice/val_dice"]
        layout_dice_ET_merge = ["dice_ET/train_dice_ET", "dice_ET/val_dice_ET"]
        layout_dice_TC_merge = ["dice_TC/train_dice_TC", "dice_TC/val_dice_TC"]
        layout_dice_WT_merge = ["dice_WT/train_dice_WT", "dice_WT/val_dice_WT"] 
        layout_dice_p_cls_merge = ["dice_p_cls/train_dice_p_cls", "dice_p_cls/val_dice_p_cls"]
        layout_mse_merge = ["mse_score/train_mse_score", "mse_score/val_mse_score"]
        layout_fmse_merge = ["fmse_score/train_fmse_score", "fmse_score/val_fmse_score"]
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
            "MSE Scores": {
                "mse_score": ["Multiline", layout_mse_merge],
                "fmse_score": ["Multiline", layout_fmse_merge],
            },
            # "assd_merge": {
            #     "assd": ["Multiline", layout_assd_merge],
            # },
        }
        tb.add_custom_scalars(layout)        

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch):
        torch.set_grad_enabled(True)
        losses, logits, masks, preds = self._shared_step(batch)
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
            #Brats Dice Scores
            self.log("dice_TC/train_dice_TC", metrics["dice_TC"], on_epoch=True)
            self.log("dice_ET/train_dice_ET", metrics["dice_ET"], on_epoch=True)
            self.log("dice_WT/train_dice_WT", metrics["dice_WT"], on_epoch=True)
            #MSE
            self.log("mse_score/train_mse_score", metrics["mse_score"], on_epoch=True)
            self.log("fmse_score/train_fmse_score", metrics["fmse_score"], on_epoch=True)

            self.logger.experiment.add_text("train_dice_p_cls", str(metrics["dice_p_cls"].tolist()), self.current_epoch)

            for i, score in enumerate(metrics["dice_p_cls"]):
                self.logger.experiment.add_scalar(f"dice_p_cls/train_dice_p_cls_{i}", score, self.current_epoch)
            #self.log("assd/train_assd", metrics["assd"], on_epoch=True)
        self.train_step_outputs.clear()
    
    def validation_step(self, batch):
        losses, logits, masks, preds = self._shared_step(batch, detach=True)
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

            #Brats Dice Scores
            self.log("dice_TC/val_dice_TC", metrics["dice_TC"], on_epoch=True)
            self.log("dice_ET/val_dice_ET", metrics["dice_ET"], on_epoch=True)
            self.log("dice_WT/val_dice_WT", metrics["dice_WT"], on_epoch=True)
            #MSE
            self.log("mse_score/val_mse_score", metrics["mse_score"], on_epoch=True)
            self.log("fmse_score/val_fmse_score", metrics["fmse_score"], on_epoch=True)

            self.logger.experiment.add_text("val_dice_p_cls", str(metrics["dice_p_cls"].tolist()), self.current_epoch)

            for i, score in enumerate(metrics["dice_p_cls"]):
                self.logger.experiment.add_scalar(f"dice_p_cls/val_dice_p_cls_{i}", score, self.current_epoch)
            #self.log("assd/val_assd", metrics["assd"], on_epoch=True)
        self.val_step_outputs.clear()

    def configure_optimizers(self):
        #optimizer = torch.optim.Adam(self.parameters(), lr=self.start_lr)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.start_lr, weight_decay=self.l2_reg_w)
        scheduler = lr_scheduler.LinearLR(
            optimizer=optimizer, start_factor=1.0, end_factor=self.linear_end_factor, total_iters=self.epochs
        )
        # scheduler = lr_scheduler.CosineAnnealingLR(
        #     optimizer, T_max=self.epochs, eta_min=0
        # )
        if scheduler is not None:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return {"optimizer": optimizer}
        
    def loss(self, logits, masks, soft_masks):
                    
        ### Regression Losses (SOFT LOSSES) ###
        if self.soft_loss_w == 0 or self.mse_loss_w == 0:
            mse_loss = torch.tensor(0.0)
        else:
            mse_loss = self.MSE(logits, soft_masks) * self.mse_loss_w * self.soft_loss_w

        if self.adw_loss_w == 0 or self.soft_loss_w == 0:
            adw_loss = torch.tensor(0.0)
        else:
            adw_loss = self.ADWL(logits, soft_masks) * self.adw_loss_w * self.soft_loss_w
        
        ### END of Regression Losses (SOFT LOSSES) ###

        #### Classification Losses (HARD LOSSES) ###
        if self.ce_loss_w == 0 or self.hard_loss_w == 0:
            ce_loss = torch.tensor(0.0)
        else:
            ce_loss = self.CEL(logits, masks.squeeze(1)) * self.ce_loss_w * self.hard_loss_w    # underlying NLLLoss can't handle mixed precision on CPU I guess (does mixed precision exist on cpu?)

        #Brats Dice Losses for subregions equally weighted
        if self.dsc_loss_w == 0 or self.hard_loss_w == 0:
            dice_ET_loss = dice_TC_loss = dice_WT_loss = torch.tensor(0.0)
        else:
            if self.binary:
                dice_ET_loss = dice_TC_loss = torch.tensor(0.0)
                
                dice_WT_loss = self.MonaiDiceBratsLoss(logits, masks) * self.dsc_loss_w * self.hard_loss_w
            
            else:
                logits_ET_FG = logits[:,[3]].sum(dim=1, keepdim=True)
                logits_ET_BG = logits[:,[0,1,2]].sum(dim=1, keepdim=True)
                
                logits_TC_FG = logits[:,[1, 3]].sum(dim=1, keepdim=True)
                logits_TC_BG = logits[:,[0, 2]].sum(dim=1, keepdim=True)

                logits_WT_FG = logits[:,[1,2,3]].sum(dim=1, keepdim=True)
                logits_WT_BG = logits[:,[0]].sum(dim=1, keepdim=True)

                logits_ET = torch.cat((logits_ET_BG, logits_ET_FG),dim=1)
                logits_TC = torch.cat((logits_TC_BG, logits_TC_FG),dim=1)
                logits_WT = torch.cat((logits_WT_BG, logits_WT_FG),dim=1)

                gt_ET_FG = (masks == 3).long()
                gt_TC_FG = (((masks == 1) | (masks == 3))).long()
                gt_WT_FG = (masks > 0).long()

                dice_ET_loss = self.MonaiDiceBratsLoss(logits_ET, gt_ET_FG) * self.dsc_loss_w * self.hard_loss_w
                dice_TC_loss = self.MonaiDiceBratsLoss(logits_TC, gt_TC_FG) * self.dsc_loss_w * self.hard_loss_w
                dice_WT_loss = self.MonaiDiceBratsLoss(logits_WT, gt_WT_FG) * self.dsc_loss_w * self.hard_loss_w

        # Brats Soft Dice Losses for subregions equally weighted
        if self.soft_dice_loss_w == 0:
            soft_dice_ET_loss = soft_dice_WT_loss = soft_dice_TC_loss = torch.tensor(0.0)
        else:
            if self.binary:
                dice_ET_loss = dice_TC_loss = torch.tensor(0.0)

                gt_WT = F.one_hot(masks, num_classes=2).permute(0, 4, 1, 2, 3)

                soft_dice_WT_loss = (1 - self.SoftDice(logits, gt_WT)) * self.soft_dice_loss_w * self.hard_loss_w
            else:
                logits_ET_FG = logits[:,[3]].sum(dim=1, keepdim=True)
                logits_ET_BG = logits[:,[0,1,2]].sum(dim=1, keepdim=True)
                
                logits_TC_FG = logits[:,[1, 3]].sum(dim=1, keepdim=True)
                logits_TC_BG = logits[:,[0, 2]].sum(dim=1, keepdim=True)

                logits_WT_FG = logits[:,[1,2,3]].sum(dim=1, keepdim=True)
                logits_WT_BG = logits[:,[0]].sum(dim=1, keepdim=True)

                logits_ET = torch.cat((logits_ET_BG, logits_ET_FG),dim=1)
                logits_TC = torch.cat((logits_TC_BG, logits_TC_FG),dim=1)
                logits_WT = torch.cat((logits_WT_BG, logits_WT_FG),dim=1)

                gt_ET_FG = (masks == 3).long().squeeze(1)
                gt_TC_FG = (((masks == 1) | (masks == 3))).long().squeeze(1)
                gt_WT_FG = (masks > 0).long().squeeze(1)

                gt_ET = F.one_hot(gt_ET_FG, num_classes=2).permute(0, 4, 1, 2, 3)
                gt_TC = F.one_hot(gt_TC_FG, num_classes=2).permute(0, 4, 1, 2, 3)
                gt_WT = F.one_hot(gt_WT_FG, num_classes=2).permute(0, 4, 1, 2, 3)

                soft_dice_ET_loss = (1 - self.SoftDice(logits_ET, gt_ET)) * self.soft_dice_loss_w * self.hard_loss_w
                soft_dice_TC_loss = (1 - self.SoftDice(logits_TC, gt_TC)) * self.soft_dice_loss_w * self.hard_loss_w
                soft_dice_WT_loss = (1 - self.SoftDice(logits_WT, gt_WT)) * self.soft_dice_loss_w * self.hard_loss_w

        # # # Weight Regularization
        # # l2_reg = torch.tensor(0.0, device=self.device).to(non_blocking=True)

        # # for param in self.model.parameters():
        # #     l2_reg += torch.norm(param).to(self.device, non_blocking=True)

        # l2_reg = torch.tensor(0.0)    

        return {
            "adw_loss": adw_loss,
            "mse_loss": mse_loss,
            "ce_loss": ce_loss,
            "soft_dice_ET_loss": soft_dice_ET_loss,
            "soft_dice_TC_loss": soft_dice_TC_loss,
            "soft_dice_WT_loss": soft_dice_WT_loss,
            "dice_ET_loss": dice_ET_loss,
            "dice_TC_loss": dice_TC_loss,
            "dice_WT_loss": dice_WT_loss,
            # "l2_reg_loss": (l2_reg * self.l2_reg_w),
        }
    
    def _loss_merge(self, losses: dict):
        return sum(losses.values())
    
    def _shared_step(self, batch, detach: bool =False):
        imgs = batch['img']     # unpacking the batch
        masks = batch['seg']    # unpacking the batch
        soft_masks = batch['soft_seg']  # unpacking the batch
        logits = self.model(imgs)   # this implicitly calls the forward method
    
        del imgs # delete the images tensor to free up memory
        
        if self.final_activation == "softmax":
            probs = self.softmax(logits)    # applying softmax to the logits to get probabilites

        elif self.final_activation == "relu":
            if bool(self.relu(logits).max()): # checking if the max value of the relu is not zero
                probs = self.relu(logits)/self.relu(logits).max()
            else: 
                probs = self.relu(logits)

        preds = torch.argmax(probs, dim=1)   # getting the class with the highest probability -> argmax not differentiable
        del probs

        if detach:
            masks = masks.detach()
            soft_masks = soft_masks.detach()
            logits = logits.detach()
            preds = preds.detach()

        losses = self.loss(logits, masks, soft_masks)

        # #printing for every param if requires_grad is True
        # for name, param in self.model.named_parameters():
        #     print(name, param.requires_grad)

        # #printing when requires_grad is False -> no backpropagation
        # for k, v in losses.items():
        #     if v is not isinstance(v, bool):
        #         if v.requires_grad == False and v != 0:
        #             print(f"{k} = {v} requires grad:")
        #             print(f"{v.requires_grad}")

        return losses, logits, masks, preds
    
    def _shared_metric_step(self, loss, logits, masks, preds):
        # if self.one_hot and not self.soft:
        #     masks = torch.argmax(masks, dim=1).long()  # convert one-hot encoded masks to integer masks

        # No squeezing of masks necessary as mF.dice implicitly squeezes dimension of size 1 (except batch size)
        # Overall Dice Scores
        dice = self.Dice(preds, masks)
        diceFG = self.DiceFG(preds, masks)
        dice_p_cls = mF.dice(preds, masks, average=None, num_classes=self.out_channels) # average=None returns dice per class

        # in the case that a class is not available in ground truth and prediction, the dice_p_cls would be NaN -> set it to 1, as it correctly predicts the absence
        for idx, score in enumerate(dice_p_cls):
            if score.isnan():
                dice_p_cls[idx] = 1.0

        if self.binary:
            dice_ET = dice_TC = torch.tensor(0.0)   # ET and TC Region Metrics don't work for binary case

            dice_WT = self.DiceFG((preds > 0), (masks > 0)) # WT (Whole Tumor) for binary case
        else:
            # ET (Enhancing Tumor): label 3
            dice_ET = self.DiceFG((preds == 3), (masks == 3))

            # TC(Tumor Core): ET + NCR = label 1 + label 3
            dice_TC = self.DiceFG((preds == 1) | (preds == 3), (masks == 1) | (masks == 3))

            # WT (Whole Tumor): TC + ED = label 1 + label 2 + label 3
            dice_WT = self.DiceFG((preds > 0), (masks > 0))

        mse_score = mF.mean_squared_error(preds, masks.squeeze(1))

        fore_mask = (masks>0).long().squeeze(1)

        fore_area = torch.sum(fore_mask).item()
        #img_area = preds.view(-1).shape[0]

        if fore_area > 0:  # Avoid division by zero
            fmse_score = mF.mean_squared_error(preds * fore_mask, masks.squeeze(1) * fore_mask) * self.img_area / fore_area
        else:
            fmse_score = torch.tensor(0.0)  # Handle the case where there is no foreground

        return {
            "loss": loss.detach(), 
            "dice": dice.detach(), 
            "diceFG": diceFG.detach(), 
            "dice_p_cls": dice_p_cls.detach(), 
            "dice_ET": dice_ET.detach(),
            "dice_TC": dice_TC.detach(),
            "dice_WT": dice_WT.detach(),
            "mse_score": mse_score.detach(),
            "fmse_score": fmse_score.detach(),
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
    
    
    def _prepare_region_logits(self, logits, target_classes):
        """
        Efficiently sum the logits for the given target classes.
        """
        return logits[:, target_classes].sum(dim=1, keepdim=True)