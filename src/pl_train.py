import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from pl_unet import LitUNetModule
from data.bids_dataset import BidsDataModule, brats_keys
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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


if __name__ == '__main__':
    # Set device automatically handled by PyTorch Lightning
    data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge'
    #data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/Sample-Subset'
    n_classes = 4   # we have 4 classes (background, edema, non-enhancing tumor, enhancing tumor)
    out_channels = n_classes    # as we don't have intermediate feature maps, our output are the final class predictions
    img_key = brats_keys[0]
    do2D = False     # Use slices and 2D Unet or whole MRI and 3D Unet
    contrast = 'multimodal'

    if contrast == 'multimodal':
        in_channels = 4
    else:
        in_channels = 1

    if n_classes == 2:
        binary = True
    else:
        binary = False

    # Hyperparameters
    start_lr = 0.0001   # starting learning rate
    lr_end_factor = 0.01 # factor to reduce learning rate to at the end of training
    l2_reg_w = 0.001    # weight for L2 regularization
    dsc_loss_w = 1.0    # weight for Dice loss
    batch_size = 1     # batch size
    max_epochs = 800    # number of epochs to train
    dim = 16    # number of filters in the first layer (has to be divisible by number of groups)
    groups = 8  # number of groups for group normalization 
    resize = (200, 200, 152) # resize the input images to this size
    accumulate_grad_batches = 4

    augmentations = Compose(
        [   
            #RandAdjustContrastd(keys=img_key, prob=0.8, gamma=(0.7,1.5)),
            #RandRotated(keys=brats_keys, range_x=math.radians(30), range_y=math.radians(30), range_z=math.radians(30), prob=0.75,keep_size=True, mode =["bilinear", "nearest"]),
            #RandAffined(keys=brats_keys, prob=0.75, translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1), mode =["bilinear", "nearest"]),
            #RandGaussianNoised(keys=img_key, prob=0.1, mean=0.0, std=0.1),
            RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=0),
            RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=1),
            RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=2),
            RandScaleIntensityd(keys=img_key, factors=0.1, prob=1.0),
            RandShiftIntensityd(keys=img_key, offsets=0.1, prob=1.0),
        ]
    )
    augmentations = None

    model = LitUNetModule(
        in_channels = in_channels,
        out_channels = out_channels,
        dim = dim,
        groups = groups,
        do2D = do2D,
        binary= binary, 
        start_lr = start_lr,
        lr_end_factor = lr_end_factor,
        n_classes = n_classes,
        l2_reg_w = l2_reg_w,
        epochs = max_epochs,
    )  

    data_module = BidsDataModule(
        data_dir = data_dir,
        contrast = contrast,
        do2D = do2D,
        binary = binary,
        batch_size = batch_size,
        train_transform = augmentations,
        resize = resize,
        test_transform=None,
    )
    
    if do2D:
        model_name='2D_UNet'
    else:
        model_name='3D_UNet'

    
    if augmentations == None:
        suffix = str(f"_batch_size_{batch_size}_n_epochs_{max_epochs}_dimUNet_{dim}_binary:{binary}_no_augmentations")
    else:
        suffix = str(f"_batch_size_{batch_size}_n_epochs_{max_epochs}_dimUNet_{dim}_binary:{binary}_with_augmentations")

    #suffix = suffix + "_DEBUGGING_RUN"

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
        version=str(f"{model_name}_v{version}_lr{start_lr}{suffix}"), # naming is a bit wack, improve later
        default_hp_metric=False,
    )

    # Profiler to monitor performance
    dirpath = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/profiler_logs'
    #profiler = pl.profilers.AdvancedProfiler(dirpath=dirpath, filename='profiler_logs')
    profiler = pl.profilers.SimpleProfiler(dirpath=dirpath, filename='performance_logs')

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')

    # Trainer handles training loops∫s∫
    trainer = pl.Trainer(
        #fast_dev_run=True,
        max_epochs=max_epochs, 
        default_root_dir='/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs', 
        log_every_n_steps=10, 
        accelerator='auto',
        logger=logger,
        callbacks=lr_monitor,
        #profiler=profiler,
        profiler='simple',
        accumulate_grad_batches=accumulate_grad_batches,
    )


    # Train the model
    trainer.fit(model, datamodule=data_module)