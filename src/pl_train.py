import os
import argparse
import lightning.pytorch as pl
from argparse import Namespace
from lightning.pytorch.callbacks import LearningRateMonitor
from pl_unet import LitUNetModule
from data.bids_dataset import BidsDataModule, brats_keys
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from monai.transforms import (
    Compose,
    RandAdjustContrastd,
    RandRotated,
    RandAffined,
    RandGaussianNoised,
    RandFlipd,
    SpatialPadd,
    CastToTyped,
    ToTensord,
    RandScaleIntensityd,
    RandShiftIntensityd,
)
import numpy as np
import torch
import math

def parse_train_param(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("-bs", type=int, default=1, help="Batch size")
    parser.add_argument("-epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("-n_cpu", type=int, default=20, help="Number of cpu workers")   # I have 20 CPU Cores
    parser.add_argument("-groups", type=int, default=8, help="Number of groups for group normalization")
    parser.add_argument("-dim", type=int, default=16, help="Number of filters in the first layer (has to be divisible by number of groups)")
    parser.add_argument("-n_accum_grad_batch", type=int, default=1, help="Number of batches to accumulate gradient over")
    #
    # action=store_true means that if the argument is present, it will be set to True, otherwise False
    #parser.add_argument("-drop_last_val", action="store_true", default=False, help="drop last in validation set during training")
    #
    parser.add_argument("-do2D", action="store_true", default=False, help="Use 2D Unet instead of 3D Unet")
    parser.add_argument("-resize", type=int, nargs= "+", default=(152, 192, 144), help="Resize the input images to this size (divisible by 8)")
    parser.add_argument("-contrast", type=str, default='multimodal', help="Type of MRI images to be used")
    parser.add_argument("-soft", action="store_true", default=False, help="Use soft segmentation masks for training")
    parser.add_argument("-one_hot", action="store_true", default=False, help="Use one-hot encoding for the labels")
    parser.add_argument("-dilation", type=int, default=0, help="Number of voxel neighbor layers to dilate to for soft masks")
    #
    parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate of the network")
    parser.add_argument("-lr_end_factor", type=float, default=0.01, help="Linear End Factor for StepLR")
    parser.add_argument("-l2_reg_w", type=float, default=0.001, help="L2 Regularization Weight Factor")
    parser.add_argument("-dsc_loss_w", type=float, default=1.0, help="Dice Loss Weight Factor")
    parser.add_argument("-ce_loss_w", type=float, default=1.0, help="Cross Entropy Loss Weight Factor")
    parser.add_argument("-soft_loss_w", type=float, default=0.0, help="Soft Loss Weight Factor")
    parser.add_argument("-sigma", type=float, default=0.125, help="Sigma for Gaussian Noise")
    #
    # action=store_true means that if the argument is present, it will be set to True, otherwise False
    parser.add_argument("-test_run", action="store_true", default=False, help="Test run with small batch size and sample size")
    parser.add_argument("-no_augmentations", action="store_true", default=False, help="No augmentations during training")
    parser.add_argument("-suffix", type=str, default="", help="Sets a setup name suffix for easier identification")
    #parser.add_argument("-gpus", type=int, default=[0], nargs="+", help="Which GPU indices are used")
    return parser

def get_option(opt: Namespace, attr: str, default, dtype: type = None):
    # option exists
    if opt is not None and hasattr(opt, attr):
        option = getattr(opt, attr)
        # dtype is given, cast to it
        if dtype is not None:
            if dtype == bool and isinstance(option, str):
                option = option in ["true", "True", "1"]
            return dtype(option)
        return option

    # option does not exist, return default
    return default

#def get_trainer_callbacks(train_loader, opt, bestf1: bool = True):
def get_trainer_callbacks(bestf1: bool = True):
    callbacks = []

    # Define save best 5 checkpoints
    mc_best = ModelCheckpoint(
        filename="epoch={epoch}-step={step}-train_loss={loss/train_loss:.4f}_train-best",
        monitor="loss/train_loss",
        mode="min",
        save_top_k=2,
        save_weights_only=True,
        verbose=False,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )
    mc_valbest = ModelCheckpoint(
        filename="epoch={epoch}-step={step}-val_loss={loss/val_loss:.4f}_val-best",
        monitor="loss/val_loss",
        mode="min",
        save_top_k=2,
        save_weights_only=True,
        verbose=False,
        save_on_train_epoch_end=True,
        auto_insert_metric_name=False,
    )
    callbacks.append(mc_best)
    callbacks.append(mc_valbest)
    if bestf1:
        mc_valbestweights = ModelCheckpoint(
            filename="epoch={epoch}-step={step}-val_diceFG={diceFG/val_diceFG:.4f}_valdiceFG-weights",
            monitor="diceFG/val_diceFG",
            mode="max",
            save_top_k=2,
            save_weights_only=True,
            verbose=False,
            save_on_train_epoch_end=True,
            auto_insert_metric_name=False,
        )
        callbacks.append(mc_valbestweights)

    # Save latest two checkpoints for safety
    mc_last = ModelCheckpoint(
        filename="epoch={epoch}-step={step}-train_loss={loss/train_loss:.4f}_latest",
        monitor="step",
        mode="max",
        #every_n_train_steps=min(200, len(train_loader)),   # train_loader is not available directly when working with DataModule
        every_n_train_steps=200,                             #hardcoded for now, fix later
        save_top_k=2,
        auto_insert_metric_name=False,
    )
    callbacks.append(mc_last)

    assert len(callbacks) > 0, "no callbacks defined"
    return callbacks


if __name__ == '__main__':

    pl.seed_everything(42)

    parser = parse_train_param()
    opt = parser.parse_args()

    if opt.soft:
        opt.one_hot = True
        opt.soft_loss_w = 1.0
        opt.ce_loss_w = 0.0
        opt.dsc_loss_w = 0.0

    print("Train with arguments")
    print(opt)
    print()

    # Set device automatically handled by PyTorch Lightning
    data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge'
    #data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/Sample-Subset'
    n_classes = 4   # we have 4 classes (background, edema, non-enhancing tumor, enhancing tumor)
    out_channels = n_classes    # as we don't have intermediate feature maps, our output are the final class predictions
    img_key = brats_keys[0]

    if opt.contrast == 'multimodal':
        in_channels = 4
    else:
        in_channels = 1

    if n_classes == 2:
        binary = True
    else:
        binary = False


    augmentations = Compose(
        [   
            #RandAdjustContrastd(keys=img_key, prob=0.5, gamma=(0.7,1.5)),
            RandRotated(keys=brats_keys, range_x=math.radians(30), range_y=math.radians(30), range_z=math.radians(30), prob=0.3,keep_size=True, mode =["bilinear", "nearest"]),
            #RandAffined(keys=brats_keys, prob=0.75, translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1), mode =["bilinear", "nearest"]),
            RandGaussianNoised(keys=img_key, prob=0.1, mean=0.0, std=0.1),
            RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=2),
            #RandScaleIntensityd(keys=img_key, factors=0.1, prob=1.0),
            #RandShiftIntensityd(keys=img_key, offsets=0.1, prob=1.0),
        ]
    )

    if opt.no_augmentations:
        augmentations = None

    model = LitUNetModule(
        conf = opt,
        in_channels = in_channels,
        out_channels = out_channels,
        epochs = opt.epochs,
        dim = opt.dim,
        groups = opt.groups,
        do2D = opt.do2D,
        binary= binary,
        soft = opt.soft,
        one_hot = opt.one_hot,
        sigma = opt.sigma,
        start_lr = opt.lr,
        lr_end_factor = opt.lr_end_factor,
        n_classes = n_classes,
        l2_reg_w = opt.l2_reg_w,
        dsc_loss_w = opt.dsc_loss_w,
        ce_loss_w = opt.ce_loss_w,
        soft_loss_w = opt.soft_loss_w
    )  

    data_module = BidsDataModule(
        data_dir = data_dir,
        contrast = opt.contrast,
        do2D = opt.do2D,
        binary = binary,
        soft = opt.soft,
        one_hot = opt.one_hot,
        batch_size = opt.bs,
        train_transform = augmentations,
        resize = tuple(opt.resize),
        test_transform=None,
        n_workers=opt.n_cpu,
    )
    
    ## Set up Model Name
    model_name='3D_UNet'
    n_aug = str("_w_augs")  # augmentations
    n_oh = str("")                      # one-hot encoding -> flag became obsolete
    n_soft = str("")                    # soft segmentation
    n_bin = str("")                     # binary segmentation
    n_sigma = str("")                   # sigma for Gaussian Noise
    n_grad_accum = str("")              # gradient accumulation

    if opt.do2D:
        model_name='2D_UNet'
        
    if augmentations == None:
        n_aug = str("_no_augs")

    # parameter one_hot became obsolete, as independent of this, the soft masks are always created     
    # if opt.one_hot:
    #     n_oh = str("_one_hot")

    if opt.soft:
        n_soft = str("_soft")
        n_sigma = str(f"_sigma_{opt.sigma}")

    if binary:
        n_bin = str("_binary")

    if opt.n_accum_grad_batch > 1:
        n_grad_accum = str(f"_grad_accum_{opt.n_accum_grad_batch}")


    suffix = str(f"_bs_{opt.bs}_epochs_{opt.epochs}_dim_{opt.dim}{n_grad_accum}{n_sigma}{n_bin}{n_oh}{n_soft}{n_aug}")
    n_loss_w = str(f"_dsc_{opt.dsc_loss_w}_ce_{opt.ce_loss_w}_soft_{opt.soft_loss_w}")

    #Directory for logs
    filepath_logs = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs'

    # Determine next version number
    #model_name = model.__class__.__name__
    log_dir = os.path.join(filepath_logs, model_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    version = len(os.listdir(log_dir))


    # Initalize logger
    logger = TensorBoardLogger(
        save_dir=str(filepath_logs),
        name=model_name,
        version=str(f"{model_name}_v{version}_lr{opt.lr}{n_loss_w}{suffix}{opt.suffix}"), # naming is a bit wack, improve later
        default_hp_metric=False,
    )

    # Profiler to monitor performance
    dirpath = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/profiler_logs'
    #profiler = pl.profilers.AdvancedProfiler(dirpath=dirpath, filename='profiler_logs')
    profiler = pl.profilers.SimpleProfiler(dirpath=dirpath, filename='performance_logs')

    # Checkpoint callback
    callbacks = get_trainer_callbacks()

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Callbacks
    callbacks.append(lr_monitor)

    # Trainer handles training loops
    trainer = pl.Trainer(
        fast_dev_run=opt.test_run,
        max_epochs=opt.epochs, 
        default_root_dir='/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs', 
        log_every_n_steps=10, 
        accelerator='auto',
        logger=logger,
        callbacks=callbacks, 
        #profiler=profiler,
        profiler='simple',
        accumulate_grad_batches=opt.n_accum_grad_batch,
    )


    # Train the model
    trainer.fit(model, datamodule=data_module)