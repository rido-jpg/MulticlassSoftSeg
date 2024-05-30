import os
import lightning.pytorch as pl
from pl_unet import LitUNet2DModule
from data.bids_dataset import BidsDataModule
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == '__main__':
    # Set device automatically handled by PyTorch Lightning
    data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge'
    in_channels = 1 # MRI scans are grayscale -> 1 channel
    n_classes = 4   # we have 4 classes (background, edema, non-enhancing tumor, enhancing tumor)
    out_channels = n_classes    # as we don't have intermediate feature maps, our output are the final class predictions

    if n_classes == 2:
        binary = True
    else:
        binary = False

    # Hyperparameters
    start_lr = 0.0001   # starting learning rate
    lr_end_factor = 0.1 # factor to reduce learning rate to at the end of training
    l2_reg_w = 0.001    # weight for L2 regularization
    dsc_loss_w = 1.0    # weight for Dice loss
    batch_size = 4
    max_epochs = 30    # number of epochs to train


    model = LitUNet2DModule(
        in_channels=1,
        out_channels=out_channels,
        start_lr=start_lr,
        lr_end_factor=lr_end_factor,
        n_classes=n_classes,
        l2_reg_w=l2_reg_w,
    )  
    data_module = BidsDataModule(
        data_dir = data_dir,
        binary = binary,
        batch_size = batch_size,
        transform = None,
    )

    suffix = str(f"_batch_size_{batch_size}_n_epochs_{max_epochs}_torchmetrics_dice_loss")

    #Directory for logs
    filepath_logs = os.getcwd() + "/lightning_logs/"

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

    # Trainer handles training loop
    trainer = pl.Trainer(
        fast_dev_run=True,
        max_epochs=max_epochs, 
        default_root_dir='/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/', 
        log_every_n_steps=1, 
        accelerator='auto',
        logger=logger,
        #profiler="simple"
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)