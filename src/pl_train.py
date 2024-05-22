import os
import lightning.pytorch as pl
from pl_unet import LitUNet2DModule, BidsDataModule
from lightning.pytorch.loggers import TensorBoardLogger

if __name__ == '__main__':
    # Set device automatically handled by PyTorch Lightning
    data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
    # Non-Binary
    model = LitUNet2DModule(in_channels=1, out_channels=4, n_classes=4, l2_reg_w=0.001, start_lr=0.0001)    # For binary -> output_channels=2 and n_classes=2
    data_module = BidsDataModule(data_dir, binary=False, batch_size=20) # For binary -> binary=True

    # Binary
    # model = LitUNet2DModule(in_channels=1, out_channels=4, n_classes=4, l2_reg_w=0.001, start_lr=0.0001)    # For binary -> output_channels=2 and n_classes=2
    # data_module = BidsDataModule(data_dir, binary=False, batch_size=20) # For binary -> binary=True

    suffix = str(f"_batch_size_{data_module.batch_size}_binary_{data_module.binary}_total_iters=20")

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
        version=str(f"{model_name}_v{version}_lr{model.start_lr}{suffix}"), # naming is a bit wack, improve later
        default_hp_metric=False,
    )

    # Trainer handles training loop
    trainer = pl.Trainer(
        max_epochs=20, 
        default_root_dir='/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/', 
        log_every_n_steps=1, 
        accelerator='auto',
        logger=logger
    )
    
    # Train the model
    trainer.fit(model, datamodule=data_module)