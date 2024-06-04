import os
import lightning.pytorch as pl
from pl_unet import LitUNet2DModule
from data.bids_dataset import BidsDataModule, brats_keys
from lightning.pytorch.loggers import TensorBoardLogger
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
)
import numpy as np
import torch
import math

if __name__ == '__main__':
    # Set device automatically handled by PyTorch Lightning
    data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge'
    in_channels = 1 # MRI scans are grayscale -> 1 channel
    n_classes = 4   # we have 4 classes (background, edema, non-enhancing tumor, enhancing tumor)
    out_channels = n_classes    # as we don't have intermediate feature maps, our output are the final class predictions
    img_key = brats_keys[0]

    if n_classes == 2:
        binary = True
    else:
        binary = False

    # Hyperparameters
    start_lr = 0.001   # starting learning rate
    lr_end_factor = 0.01 # factor to reduce learning rate to at the end of training
    l2_reg_w = 0.001    # weight for L2 regularization
    dsc_loss_w = 1.0    # weight for Dice loss
    batch_size = 32     # batch size
    max_epochs = 400    # number of epochs to train

    # augmentations = Compose(
    #     [   
    #         RandAdjustContrastd(keys=img_key, prob=0.8, gamma=(0.7,1.5)),
    #         RandRotated(keys=brats_keys, range_x=math.radians(30), range_y=math.radians(30), range_z=math.radians(30), prob=0.3,keep_size=True, mode =["bilinear", "nearest"]),
    #         RandAffined(keys=brats_keys, prob=0.4, translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1), mode =["bilinear", "nearest"]),
    #         RandGaussianNoised(keys=img_key, prob=0.1, mean=0.0, std=0.1),
    #         RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=0),
    #         SpatialPadd(keys=brats_keys, spatial_size=(256, 256, 256), mode="constant"),
    #         CastToTyped(keys=brats_keys, dtype=(torch.float, torch.long)),
    #         #ToTensord(keys=brats_keys),
    #     ]
    # )
    
    augmentations = Compose(
        [   
            RandAdjustContrastd(keys=img_key, prob=0.8, gamma=(0.7,1.5)),
            RandRotated(keys=brats_keys, range_x=math.radians(30), range_y=math.radians(30), range_z=math.radians(30), prob=0.75,keep_size=True, mode =["bilinear", "nearest"]),
            RandAffined(keys=brats_keys, prob=0.75, translate_range=(10, 10, 10), scale_range=(0.1, 0.1, 0.1), mode =["bilinear", "nearest"]),
            RandGaussianNoised(keys=img_key, prob=0.1, mean=0.0, std=0.1),
            RandFlipd(keys=brats_keys, prob=0.5, spatial_axis=0),
            SpatialPadd(keys=brats_keys, spatial_size=(256, 256, 256), mode="constant"),
            CastToTyped(keys=brats_keys, dtype=(torch.float, torch.long)),
            #ToTensord(keys=brats_keys),
        ]
    )

    #augmentations = None


    model = LitUNet2DModule(
        in_channels=1,
        out_channels=out_channels,
        start_lr=start_lr,
        lr_end_factor=lr_end_factor,
        n_classes=n_classes,
        l2_reg_w=l2_reg_w,
        epochs=max_epochs,
    )  
    data_module = BidsDataModule(
        data_dir = data_dir,
        binary = binary,
        batch_size = batch_size,
        train_transform = augmentations,
        test_transform=None,
    )

    suffix = str(f"_batch_size_{batch_size}_n_epochs_{max_epochs}_binary_with_augmentations")

    #Directory for logs
    #filepath_logs = os.getcwd() + "/lightning_logs/"
    filepath_logs = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs'

    # Determine next version number
    model_name = model.__class__.__name__
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

    # Trainer handles training loop
    trainer = pl.Trainer(
        #fast_dev_run=True,
        max_epochs=max_epochs, 
        default_root_dir='/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs', 
        log_every_n_steps=10, 
        accelerator='auto',
        logger=logger,
        profiler=profiler,
        #profiler='simple'
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)