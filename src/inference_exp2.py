import os
import sys
import torch
import TPTBox
import re
import csv
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import utils.fastnumpyio.fastnumpyio as fnio
import argparse
from pathlib import Path
from argparse import Namespace
from pl_unet import LitUNetModule
from TPTBox import NII
from utils.brats_tools import get_central_slice, plot_slices
from monai.transforms import ResizeWithPadOrCrop, CastToType
from torch import nn
from data.bids_dataset import modalities, create_bids_path_list_of_dicts, BidsDataset, BidsDataModule
import lightning.pytorch as pl
import time
import torchmetrics.functional as mF

# file = Path(__file__).resolve()
# sys.path.append(str(file.parents[1]))
# sys.path.append(str(file.parents[2]))

def avd_rvd(abs_predicted_vol, abs_reference_vol):
    avd = abs_predicted_vol - abs_reference_vol
    rvd = avd / abs_reference_vol
    return avd, rvd

def get_best_checkpoint(directory: str):
    # Define the directory path
    directory_path = Path(directory) / "checkpoints"
    
    # Regular expression pattern to capture val_diceFG value
    pattern = re.compile(r"val_diceFG=([0-9.]+)")

    best_ckpt = None
    best_val_diceFG = -1

    # Iterate over all .ckpt files in the checkpoints folder
    for ckpt_file in directory_path.glob("*.ckpt"):
        match = pattern.search(ckpt_file.name)
        if match:
            # Convert the captured value to a float
            val_diceFG = float(match.group(1))
            # Check if this is the highest val_diceFG so far
            if val_diceFG > best_val_diceFG:
                best_val_diceFG = val_diceFG
                best_ckpt = ckpt_file

    return best_ckpt

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

def parse_inf_param(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    
    parser.add_argument("-model_dir", type=str, default = None, help="Path to the model log folder. Script will automatically obtain the path of the best cpkt (best DiceFG)")
    parser.add_argument("-data_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/val", help="Path to the data directory")
    parser.add_argument("-save_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/preds", help="Path to the directory where the predictions should be saved")
    parser.add_argument("-eval_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval", help="Path to the directory where the eval.tsv should be saved")
    parser.add_argument("-suffix", type=str, default = None, help="Suffix for saved predictions")
    parser.add_argument("-test_loop", action='store_true', help="Run test loop with single sample for debugging")
    parser.add_argument("-format", type=str, default = "fnio", help="Format of the data (fnio or nii.gz)")
    parser.add_argument("-activation", type=str, default = "softmax", choices=["softmax", "relu", "linear"], help="Activation function for output layer")
    parser.add_argument("-round", type=int, default = None, help="Round all probability maps to the given number of decimals")
    parser.add_argument("-axis", type=str, default= "axial", choices=["axial", "sagittal", "coronal"], help="Axis to plot the slices. Annotation is made in Axial Slices. Hence Coronal and Sagittal slices should show hard edges in GT.")
    parser.add_argument("-samples", type=int, default = 0, help="Number of samples to predict. If 0, predict all samples in the dataset")

    parser.add_argument("-niftis", action='store_true', help="Save niftis of the predictions")
    parser.add_argument("-slices", action='store_true', help="Save slices of the predictions and ground truths")
    #parser.add_argument("-gts", action='store_true', help="Save slices of the ground truth")
    parser.add_argument("-soft", action='store_true', help="Save channelwise soft predictions/probabilities")


    return parser

def _extract_unet_version(path: Path) -> str:
    # Convert Path object to string
    path_str = str(path)
    
    # Regular expression to match '3D_UNet_v' followed by a number
    match = re.search(r"3D_UNet_v\d+", path_str)
    
    # Return the matched part if found, otherwise return None
    return match.group(0) if match else None

softmax = nn.Softmax(dim=1)
relu = nn.ReLU()

if __name__ == '__main__':

    torch.set_grad_enabled(False)

    t = time.process_time()
    print(f"Start Time: {t} seconds")

    parser = parse_inf_param()
    conf = parser.parse_args()

    model_path : Path = Path(conf.model_dir)
    data_dir : Path = Path(conf.data_dir)
    save_dir : Path = Path(conf.save_dir)
    eval_dir : Path = Path(conf.eval_dir)
    ckpt_path = get_best_checkpoint(model_path)

    print(f"Best checkpoint: {ckpt_path.name}")

    suffix = conf.suffix

    #save_gts = conf.gts
    save_niftis = conf.niftis
    save_slices = conf.slices

    if not save_slices and not save_niftis:
        print("Error: No output format selected. Please select at least one of the following: -niftis, -slices")
        sys.exit(1)

    test_loop = conf.test_loop
    format = conf.format
    axes = {'sagittal': 0, 'coronal': 1, 'axial': 2}
    axis = axes[conf.axis]
    samples = conf.samples
    if test_loop:
        samples = 1
        
    
    # try if torch.load works, otherwise raise an error and exit
    try:
        checkpoint = torch.load(ckpt_path)
    except Exception as e:
        print(f"Error: Invalid checkpoint path or file not found: {ckpt_path}")
        sys.exit(1)

    # load hparams from checkpoint to get contrast
    hparams = LitUNetModule.load_from_checkpoint(ckpt_path).hparams
    train_opt = hparams.get('opt')

    experiment = get_option(train_opt, 'experiment', 1) # if no experiment indicator given, it was experiment 1

    contrast = get_option(train_opt, 'contrast', 't2f')  # if the contrast is not part of hparams, it is an old ckpt which used 't2f'
    
    if suffix is None:
        suffix = get_option(train_opt,'suffix', '')

    model_name = f"{_extract_unet_version(ckpt_path)}_{suffix}"
    print(f"Model name: {model_name}")

    model_input_size = get_option(train_opt, 'resize', [200, 200, 152])

    print(f"Model input size: {model_input_size}")

    og_img_size = [240,240,155]

    resize = ResizeWithPadOrCrop(og_img_size)
    #crop = ResizeWithPadOrCrop(model_input_size)

    ###### CREATING TSV FOR SAVING THE SOFT METRICS
    output_file = os.path.join(eval_dir, f"{model_name}.tsv")
    
    # Define the column names
    columns = [
    "subject_name", 
    "mse_soft", "mae_soft",
    "mse_hard", "mae_hard", 
    "avd_soft", "rvd_soft", 
    "avd_soft_hard", "rvd_soft_hard", 
    "avd_hard", "rvd_hard"
    ]


    ###### CREATING TSV FOR SAVING THE SOFT METRICS


    # load trained model
    model = LitUNetModule.load_from_checkpoint(ckpt_path)
    model.eval()

    # # create BidsDataModule instance
    # bids_dm = BidsDataModule(train_opt, str(data_dir))
    # bids_dm.setup()

    # val_dataloader = bids_dm.val_dataloader()
    # trainer = pl.Trainer(inference_mode=True)

    # logits = trainer.predict(model,val_dataloader)

    # print(logits.shape)

    # create BidsDataset instance
    bids_val_ds = BidsDataset(train_opt, data_dir)

    # Open the .tsv file for writing
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter='\t')
        writer.writeheader()  # Write the header row

        for idx, dicts in enumerate(bids_val_ds):
            img = dicts['img']      # shape [4, 80, 96, 72]
            gt = dicts['seg']       # shape [1, 80, 96, 72]
            soft_gt = dicts['soft_seg'] #shape [2, 80, 96, 72]

            gt_vol = gt.sum()     # just sum all ones
            soft_gt_vol = soft_gt[1].sum() # sum all float values along the foreground dimension (channel 1)

            subject = bids_val_ds.bids_list[idx]['subject']

            print(f"---------------------------------")
            print(f"Subject: {subject}")

            # add batch dimension, in this case 1 because our batch size is 1
            img_tensor = img.unsqueeze(0)

            # ensure img_tensor is on same device as trained_model
            img_tensor = img_tensor.to(next(model.parameters()).device) 

            with torch.no_grad():
                logits = model(img_tensor)  # get logits from model

                if conf.activation == 'softmax':
                    probs = softmax(logits) # apply softmax to get probabilities
                elif conf.activation == 'relu':
                    if bool(relu(logits).max()): # checking if the max value of the relu is not zero
                        probs = relu(logits)/relu(logits).max()
                    else: 
                        probs = relu(logits)
                if conf.activation == 'linear':
                    probs = logits

                preds = torch.argmax(probs, dim=1) # get class with highest probability
                preds_cpu = preds.cpu() # move tensor to cpu; shape [1, 80, 96, 72]
                probs_cpu = probs.cpu() # move tensor to cpu; shape [1, 2, 80, 96, 72]
                probs_cpu.squeeze_(0)   # shape [2, 80, 96, 72]
                del logits, preds, probs # delete tensors to free up memory

                # del logits, preds # delete tensors to free up memory
                # if conf.soft:
                #     probs_cpu = probs.cpu() # move tensor to cpu
                # del probs

            ## EVAL OF SOFT SCORES/PROBABILITES OUTPUT BY MODEL AND NATURALLY CREATED SOFT GTs THROUGH DOWNSAMPLING
            mse_soft = mF.mean_squared_error(probs_cpu, soft_gt)
            mae_soft = mF.mean_absolute_error(probs_cpu, soft_gt)
            mse_hard = mF.mean_squared_error(preds_cpu, gt)
            mae_hard = mF.mean_absolute_error(preds_cpu, gt)
            probs_vol = probs_cpu[1].sum()    # sum of floats along foregound channel
            preds_vol = preds_cpu.sum()
            avd_soft, rvd_soft = avd_rvd(probs_vol, soft_gt_vol)
            avd_soft_hard, rvd_soft_hard = avd_rvd(probs_vol, gt_vol)   # difference between predicted soft scores and hard gt volume
            avd_hard, rvd_hard = avd_rvd(preds_vol, gt_vol)

            # Generate random data for demonstration; replace with actual calculations
            data = {
                "subject_name": subject,
                "mse_soft": mse_soft.item(),
                "mae_soft": mae_soft.item(),
                "mse_hard": mse_hard.item(),
                "mae_hard": mae_hard.item(),
                "avd_soft": avd_soft.item(),
                "rvd_soft": rvd_soft.item(),
                "avd_soft_hard": avd_soft_hard.item(),
                "rvd_soft_hard": rvd_soft_hard.item(),
                "avd_hard": avd_hard.item(),
                "rvd_hard": rvd_hard.item(),
            }

            # Write the row for the current subject
            writer.writerow(data)

            if test_loop:     
                # for testing purposes -> exit after one iteration
                break

            # base_path : Path = data_dir / subject 
            # save_path : Path = save_dir / model_name / subject / f"standard"

            # #check if future subfolder exist, else create it
            # slice_path : Path = save_path / f"slices"

            # if not slice_path.exists():
            #     slice_path.mkdir(parents=True)

            # preds_cpu = preds_cpu.type(torch.float16)    # change type of preds_cpu from torch.int64 to torch.float16        
            # preds_resized = resize(preds_cpu) # resize preds_cpu to original image size
            # preds_resized = preds_resized.squeeze(0) # get rid of batch dimension

            # preds_array = preds_resized.numpy() # convert to numpy array (dtype: float16)

            # preds_array = preds_array.astype(np.uint8) # ensure correct type to be able to cast it to a NII object

            # # load GT segmentation as NII object 
            # seg_path = base_path / f"{subject}-seg.nii.gz"
            # seg = NII.load(seg_path, seg=True)

            # pred_nii: NII = seg.set_array(preds_array)  # get prediction as nii object

            # if save_niftis:
            #     pred_nii.save(save_path/ f"nifti" / f"{subject}-pred-{suffix}.nii.gz")    # save prediction as nifti file to view it in ITK Snap

            #     # get difference between original segmentation mask and prediction
            #     difference_nifti = NII.get_segmentation_difference_to(pred_nii, seg, ignore_background_tp=True)
            #     difference_nifti.save(save_path / f"nifti" /f"{subject}-seg-difference-{suffix}.nii.gz")

            # # if save_gts:
            # #     gt_slice = get_central_slice(seg.get_array(), axis) # get central slice of ground truth
            

            # if save_slices:

            #     gt_slice = get_central_slice(seg.get_array(), axis) # get central slice of ground truth
            #     pred_slice = get_central_slice(preds_array, axis) # get central slice of prediction

            #     for contrast in modalities:
            #         if format == 'nii.gz':
            #             full_img = NII.load(bids_val_ds.bids_list[idx][contrast], seg=False)
            #             img_array = full_img.get_array()
            #         elif format == 'fnio':
            #             img_array = fnio.load(str(bids_val_ds.bids_list[idx][contrast]))

            #         img_slice = get_central_slice(img_array, axis)
            #         plot_slices(img_slice, pred_slice,plt_title='Prediction '+ suffix , save_path= save_path / f"slices" /  f"{subject}-{contrast}-{conf.axis}-slice-pred-{suffix}.png", show=False)
            #         # if save_gts:
            #         #     plot_slices(img_slice, gt_slice, plt_title='Ground Truth',save_path=save_path / f"slices" / f"{subject}-{contrast}-{conf.axis}-slice-gt.png", show = False)
            #         plot_slices(img_slice, gt_slice, plt_title='Ground Truth',save_path=save_path / f"slices" / f"{subject}-{contrast}-{conf.axis}-slice-gt.png", show = False)

            
            # if conf.soft:
            #     soft_save_path = save_dir / model_name / subject / f"channelwise"
            #     soft_slice_path : Path = soft_save_path / f"slices"

            #     if not soft_slice_path.exists():
            #         soft_slice_path.mkdir(parents=True)

            #     # load soft ground truth segmentation
            #     soft_gt = dicts['soft_seg'] 

            #     # with torch.no_grad():
            #     #     logits = model(img_tensor)  # get logits from model

            #     #     if conf.activation == 'softmax':
            #     #         probs = softmax(logits) # apply softmax to get probabilities
            #     #     elif conf.activation == 'relu':
            #     #         if bool(relu(logits).max()): # checking if the max value of the relu is not zero
            #     #             probs = relu(logits)/relu(logits).max()
            #     #         else: 
            #     #             probs = relu(logits)
            #     #     elif conf.activation == 'linear':
            #     #         probs = logits

            #     prob_channels = []
            #     gt_channels = []

            #     soft_gt = soft_gt.unsqueeze(0) # add batch dimension

            #     for channel in range(probs_cpu.shape[1]):
            #         prob_channels.append(resize(probs_cpu[:, channel])) # resize probs_cpu to original image size
            #         gt_channels.append(resize(soft_gt[:, channel]))     # resize soft_gt to original image size

            #     probs_padded = torch.stack(prob_channels, dim=1) # stack channels to get shape (1, C, H, W, D)
            #     gt_padded = torch.stack(gt_channels, dim=1) # stack channels to get shape (C, H, W, D)
                
            #     probs_padded = probs_padded.squeeze(0) # get rid of batch dimension
            #     gt_padded = gt_padded.squeeze(0) # get rid of batch dimension
                
            #     probs_array = probs_padded.numpy() # convert to numpy array (dtype: float32)
            #     gt_array = gt_padded.numpy() # convert to numpy array (dtype: float32)

            #     if conf.round:
            #         probs_array = probs_array.round(conf.round) # round probabilities to given number of decimals
            #         gt_array = gt_array.round(conf.round) # round probabilities to given number of decimals
                
            #     # load MRI as NII object
            #     sample_img_path = base_path / f"{subject}-t1c.nii.gz"
            #     nii_img = NII.load(sample_img_path,seg=False)
            #     nii_img.set_dtype_(np.float32)

            #     if format == 'nii.gz':
            #         full_img = NII.load(bids_val_ds.bids_list[idx]['t1c'], seg=False)
            #         img_array = full_img.get_array()
            #     elif format == 'fnio':
            #         img_array = fnio.load(str(bids_val_ds.bids_list[idx]['t1c']))

            #     for channel in range(probs_array.shape[0]):
            #         nii_prob: NII = nii_img.set_array(probs_array[channel])  # get prediction as nii object
            #         nii_soft_gt: NII = nii_img.set_array(gt_array[channel])  # get prediction as nii object

            #         if save_niftis:
            #             nii_prob.save(soft_save_path / f"niftis" / f"{subject}-prob-class{channel}-{suffix}.nii.gz")    # save prediction as nifti file to view it in ITK Snap
            #             nii_soft_gt.save(soft_save_path / f"niftis" / f"{subject}-soft_gt-class{channel}-sigma-{train_opt.sigma}.nii.gz")

            #         if save_slices:
            #             prob_slice = get_central_slice(probs_array[channel], axis) # get central slice of predicted probabilities
            #             gt_slice = get_central_slice(gt_array[channel], axis)      # get central slice of soft ground truth probabilities

            #             img_slice = get_central_slice(img_array, axis)
                    
            #             plot_slices(img_slice, prob_slice,plt_title=f"Predicted probability channel {channel} {suffix} ", save_path=soft_save_path / f"slices" / f"{subject}-t1c-{conf.axis}-slice-probability-class-{channel}-{suffix}.png", show=False)
            #             plot_slices(img_slice, gt_slice, plt_title=f"Soft GT probability channel {channel} sigma {train_opt.sigma}", save_path=soft_save_path / f"slices" / f"{subject}-t1c-{conf.axis}-slice-soft_gt-class-{channel}-sigma-{train_opt.sigma}.png", show=False)
    
            # if samples > 0:
            #     if idx == (samples - 1):
            #         break 

    elapsed_time = time.process_time() - t
    print(f"Process took this much time: {elapsed_time} seconds")