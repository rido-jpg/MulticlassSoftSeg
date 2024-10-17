import os
import argparse
import lightning.pytorch as pl
from argparse import Namespace
from lightning.pytorch.callbacks import LearningRateMonitor
from pl_unet import LitUNetModule
from pl_unet_cityscapes import LitUNetCityModule
from data.bids_dataset import BidsDataModule, brats_keys, modalities
from data.cityscapes_dataset import CityscapesDataModule
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

img_key = brats_keys[0]

augmentations = Compose(
    [   
        #RandAdjustContrastd(keys=img_key, prob=0.5, gamma=(0.7,1.5)),
        RandRotated(keys=brats_keys, range_x=math.radians(30), range_y=math.radians(30), range_z=math.radians(30), prob=0.3,keep_size=True, mode =["bilinear", "nearest", "nearest"]),
        #RandAffined(keys=brats_keys, prob=0.75, translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1), mode =["bilinear", "nearest"]),
        RandGaussianNoised(keys=img_key, prob=0.1, mean=0.0, std=0.1),
        RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=2),
        #RandScaleIntensityd(keys=img_key, factors=0.1, prob=1.0),
        #RandShiftIntensityd(keys=img_key, offsets=0.1, prob=1.0),
    ]
)

def parse_train_param(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("-dataset", type=str, default= 'brats', choices = ['brats', 'cityscapes'], help = 'Which dataset to train the model on')

    parser.add_argument("-bs", type=int, default=1, help="Batch size")
    parser.add_argument("-epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("-n_cpu", type=int, default=20, help="Number of cpu workers")   # I have 20 CPU Cores
    parser.add_argument("-groups", type=int, default=8, help="Number of groups for group normalization")
    parser.add_argument("-dim", type=int, default=16, help="Number of filters in the first layer (has to be divisible by number of groups)")
    parser.add_argument("-grad_accum", type=int, default=1, help="Number of batches to accumulate gradient over")
    parser.add_argument("-precision", type=str, default = 'mixed', choices=['full', 'mixed'], help = "Precision for training, full is 32-bit, mixed is 16-bit")
    parser.add_argument("-matmul_precision", type=str, default='high', choices=['highest', 'high', 'medium'], help="Precision for Matrix multiplications")
    parser.add_argument("-val_every_n_epoch", type=int, default=1, help="Validation every n epochs")

    parser.add_argument("-activation", type=str, default="softmax", choices=["softmax", "relu", "sigmoid"], help="Final activation function")
    #
    # action=store_true means that if the argument is present, it will be set to True, otherwise False
    #parser.add_argument("-drop_last_val", action="store_true", default=False, help="drop last in validation set during training")
    #
    parser.add_argument("-do2D", action="store_true", default=False, help="Use 2D Unet instead of 3D Unet")
    parser.add_argument("-resize", type=int, nargs= "+", default=(152, 192, 144), help="Resize the input images to this size (divisible by 8)")
    parser.add_argument("-contrast", type=str, default='multimodal', choices= [modalities, 'multimodal'], help="Type of MRI images to be used")
    parser.add_argument("-soft", action="store_true", default=False, help="Use soft segmentation masks and regression loss (Adaptive Wing Loss) for training")
    parser.add_argument("-one_hot", action="store_true", default=False, help="Use one-hot encoding for the labels")
    parser.add_argument("-dilate", type=int, default=0, help="Number of voxel neighbor layers to dilate to for soft masks")
    #
    parser.add_argument("-lr", type=float, default=1e-4, help="Learning rate of the network")
    parser.add_argument("-lr_end_factor", type=float, default=0.01, help="Linear End Factor for StepLR")

    parser.add_argument("-l2_reg_w", type=float, default=1e-3, help="L2 Regularization Weight Factor")
    parser.add_argument("-dsc_loss_w", type=float, default=1.0, help="Dice Loss Weight Factor")
    parser.add_argument("-ce_loss_w", type=float, default=1.0, help="Cross Entropy Loss Weight Factor")
    parser.add_argument("-hard_loss_w", type=float, default=1.0, help="Factor that all Classification Losses (CE, Dice) are multiplied with")
    parser.add_argument("-soft_loss_w", type=float, default=1.0, help="Factor that all Regression Losses (MSE, ADW) are multiplied with")
    parser.add_argument("-mse_loss_w", type=float, default=0.0, help="Mean Squared Error Loss Weight Factor")
    parser.add_argument("-adw_loss_w", type=float, default=0.0, help="Adaptive Wing Loss Weight Factor")
    parser.add_argument("-soft_dice_loss_w", type=float, default=0.0, help="Soft Dice Loss Weight Factor")

    parser.add_argument("-threshold", type=float, default=0.5, choices = np.arange(0, 1,0.1),  help="Threshold for conversion of binary probabilities to predictions")
    parser.add_argument("-sigma", type=float, default=0.125, help="Sigma for Gaussian Noise. Min value is 0.125 because otherwise the kernel radius is 0 (Radius = round(4*sigma))")
    parser.add_argument("-ds_factor", type = int, default = None, help ="What factor to downsample the images and ground truths by")
    #
    # action=store_true means that if the argument is present, it will be set to True, otherwise False
    parser.add_argument("-test_run", action="store_true", default=False, help="Test run with small batch size -> using fast_dev_run of pl.Trainer")
    parser.add_argument("-sample_subset", action="store_true", default=False, help="Test run with smaller sample subset of data")
    parser.add_argument("-min_config", action="store_true", default=False, help="Use minimal configuration for testing, e.g. small bs, dim, crop etc.")
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

    pl.seed_everything(42, workers=True)

    parser = parse_train_param()
    opt = parser.parse_args()

    if opt.soft:
        opt.one_hot = True
        opt.soft_loss_w = 1.0
        opt.adw_loss_w = 1.0
        opt.mse_loss_w = 0.0
        opt.ce_loss_w = 0.0
        opt.dsc_loss_w = 0.0
        opt.hard_loss_w = 0.0

    if opt.hard_loss_w == 0.0:
        opt.ce_loss_w = 0.0
        opt.dsc_loss_w = 0.0
    
    if opt.soft_loss_w == 0.0:
        opt.adw_loss_w = 0.0
        opt.mse_loss_w = 0.0

    if opt.min_config:
        opt.bs = 1
        opt.epochs = 5
        opt.dim = 1
        opt.groups = 1
        opt.resize = (8, 8, 8)
        opt.sample_subset = True
        opt.no_augmentations = True
        opt.n_cpu = 0

    if opt.precision == 'mixed':
        opt.precision = '16-mixed'
    else:
        opt.precision = '32-true'

    if opt.no_augmentations:
        augmentations = None

    if opt.dataset == 'brats':
        
        data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge'
        
        if opt.sample_subset:
            data_dir = data_dir + '/Sample-Subset'
        
        n_classes = 4   # we have 4 classes (background, edema, non-enhancing tumor, enhancing tumor)
        out_channels = n_classes    # as we don't have intermediate feature maps, our output are the final class predictions
        
        if opt.contrast == 'multimodal':
            in_channels = 4
        else:
            in_channels = 1

        if n_classes == 2:
            binary = True
        else:
            binary = False
        
        data_module = BidsDataModule(opt = opt, data_dir = data_dir,binary = binary, train_transform = augmentations, test_transform=None)

        model = LitUNetModule(opt = opt, in_channels = in_channels, out_channels = out_channels, binary= binary, n_classes = n_classes)

    elif opt.dataset == 'cityscapes':
        opt.do2D = True

        opt.activation = 'sigmoid'

        in_channels = 3 # rgb channels

        n_classes = 1   #  only traffic signs -> 1 output channel is enough 
        out_channels = n_classes

        binary = True

        data_module = CityscapesDataModule(opt = opt)

        model = LitUNetCityModule(opt = opt, in_channels = in_channels, out_channels = out_channels, binary= binary, n_classes = n_classes)


    print("Train with arguments")
    print(opt)
    print()

    # Set device automatically handled by PyTorch Lightning

    torch.set_float32_matmul_precision(opt.matmul_precision)

    # Compile the model
    #model = torch.compile(model)
    
    ## Set up Model Name
    model_name='3D_UNet'
    n_activ = str(f"_{opt.activation}")  # activation function
    n_aug = str("_w_augs")              # augmentations
    n_oh = str("")                      # one-hot encoding -> flag became obsolete
    n_soft = str("")                    # soft segmentation
    n_bin = str("")                     # binary segmentation
    n_sigma = str("")                   # sigma for Gaussian Noise
    n_grad_accum = str("")              # gradient accumulation
    n_dilate = str("")                  # number of voxel neighbor layers to dilate to for soft masks
    n_down = str("")                    # downsampling factor

    if opt.do2D:
        model_name='2D_UNet'
        
    if augmentations == None:
        n_aug = str("_no_augs")

    # parameter one_hot became obsolete, as independent of this, the soft masks are always created     
    # if opt.one_hot:
    #     n_oh = str("_one_hot")

    if opt.sigma != 0:
        n_sigma = str(f"_sigma_{opt.sigma}")

    if opt.ds_factor is not None:
        n_down = str(f"_down_factor_{opt.ds_factor}")

    if opt.dilate != 0:
        n_dilate = str(f"_dilate_{opt.dilate}")

    if binary:
        n_bin = str("_binary")

    if opt.grad_accum > 1:
        n_grad_accum = str(f"_grad_accum_{opt.grad_accum}")

    # include loss weights in model name if non-zero
    n_loss_w = str("")
    for arg in vars(opt).items():
        key, value = arg
        if 'loss' in key:
            substring = key.replace("_loss_w", "")
            if value != 0.0:
                n_loss_w = str(f"{n_loss_w}_{substring}_{value}")

    #suffix = str(f"_bs_{opt.bs}_epochs_{opt.epochs}_dim_{opt.dim}_precision_{opt.precision}_matmulprec_{opt.matmul_precision}_{n_loss_w}{n_grad_accum}{n_sigma}{n_dilate}{n_bin}{n_oh}{n_soft}{n_activ}{n_aug}")
    suffix = str(f"{n_loss_w}{n_grad_accum}{n_down}{n_sigma}{n_dilate}{n_bin}{n_oh}{n_soft}{n_activ}")

    #Directory for logs
    filepath_logs = f"/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs/{opt.dataset}"

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
        version=str(f"{model_name}_v{version}_lr{opt.lr}{suffix}{opt.suffix}"), # naming is a bit wack, improve later
        default_hp_metric=False,
        log_graph=False,     # to log the Model Graph
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
        check_val_every_n_epoch=opt.val_every_n_epoch,
        precision = opt.precision, 
        log_every_n_steps=10, 
        accelerator='gpu',
        devices = -1,
        detect_anomaly=True,
        logger=logger,
        callbacks=callbacks, 
        #profiler=profiler,
        profiler='simple',
        inference_mode=True,    # optimizes performance during evaluation phases, such as validation, testing, and prediction, by disabling gradient calculations -> makes using torch.no_grad obsolete
        deterministic = True,   # to ensure compatibility between runs
        accumulate_grad_batches=opt.grad_accum,
    )


    # Train the model
    trainer.fit(model, datamodule=data_module)