import os
import sys
import torch
import TPTBox
import re
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
from data.bids_dataset import modalities, create_bids_path_list_of_dicts, BidsDataset

# file = Path(__file__).resolve()
# sys.path.append(str(file.parents[1]))
# sys.path.append(str(file.parents[2]))

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
    
    parser.add_argument("-ckpt_dir", type=str, default = None, help="Path to the checkpoint file")
    parser.add_argument("-data_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/val", help="Path to the data directory")
    parser.add_argument("-save_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/preds", help="Path to the directory where the predictions should be saved")
    parser.add_argument("-suffix", type=str, default = None, help="Suffix for saved predicitons")
    parser.add_argument("-test_loop", action='store_true', help="Run test loop with single sample for debugging")
    parser.add_argument("-format", type=str, default = "fnio", help="Format of the data (fnio or nii.gz)")
    parser.add_argument("-activation", type=str, default = "softmax", choices=["softmax", "relu"], help="Activation function for output layer")
    parser.add_argument("-round", type=int, default = None, help="Round all probability maps to the given number of decimals")
    parser.add_argument("-axis", type=str, default= "axial", choices=["axial", "sagittal", "coronal"], help="Axis to plot the slices")
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

    parser = parse_inf_param()
    conf = parser.parse_args()

    ckpt_path : Path = Path(conf.ckpt_dir)
    data_dir : Path = Path(conf.data_dir)
    save_dir : Path = Path(conf.save_dir)

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
    contrast = get_option(train_opt, 'contrast', 't2f')  # if the contrast is not part of hparams, it is an old ckpt which used 't2f'
    
    if suffix is None:
        suffix = get_option(train_opt,'suffix', '')

    model_name = f"{_extract_unet_version(ckpt_path)}_{suffix}"
    print(f"Model name: {model_name}")

    model_input_size = get_option(train_opt, 'resize', [200, 200, 152])

    print(f"Model input size: {model_input_size}")

    og_img_size = [240,240,155]

    pad = ResizeWithPadOrCrop(og_img_size)
    #crop = ResizeWithPadOrCrop(model_input_size)

    # load trained model
    model = LitUNetModule.load_from_checkpoint(ckpt_path)
    model.eval()

    # create BidsDataset instance
    bids_val_ds = BidsDataset(train_opt, data_dir)

    for idx, dicts in enumerate(bids_val_ds):
        img = dicts['img']

        subject = bids_val_ds.bids_list[idx]['subject']

        print(f"---------------------------------")
        print(f"Subject: {subject}")

        base_path : Path = data_dir / subject 
        save_path : Path = save_dir / model_name / subject / f"standard"

        #check if future subfolder exist, else create it
        slice_path : Path = save_path / f"slices"

        if not slice_path.exists():
            slice_path.mkdir(parents=True)

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

            preds = torch.argmax(probs, dim=1) # get class with highest probability
            preds_cpu = preds.cpu() # move tensor to cpu
            del logits, probs, preds # delete tensors to free up memory

        preds_cpu = preds_cpu.type(torch.float16)    # change type of preds_cpu from torch.int64 to torch.float16

        preds_padded = pad(preds_cpu) # pad preds_cpu to original image size
        preds_padded = preds_padded.squeeze(0) # get rid of batch dimension

        preds_array = preds_padded.numpy() # convert to numpy array (dtype: float16)

        preds_array = preds_array.astype(np.uint8) # ensure correct type to be able to cast it to a NII object

        # load GT segmentation as NII object 
        seg_path = base_path / f"{subject}-seg.nii.gz"
        seg = NII.load(seg_path, seg=True)

        pred_nii: NII = seg.set_array(preds_array)  # get prediction as nii object

        if save_niftis:
            pred_nii.save(save_path/ f"nifti" / f"{subject}-pred-{suffix}.nii.gz")    # save prediction as nifti file to view it in ITK Snap

            # get difference between original segmentation mask and prediction
            difference_nifti = NII.get_segmentation_difference_to(pred_nii, seg, ignore_background_tp=True)
            difference_nifti.save(save_path / f"nifti" /f"{subject}-seg-difference-{suffix}.nii.gz")

        # if save_gts:
        #     gt_slice = get_central_slice(seg.get_array(), axis) # get central slice of ground truth
        

        if save_slices:

            gt_slice = get_central_slice(seg.get_array(), axis) # get central slice of ground truth
            pred_slice = get_central_slice(preds_array, axis) # get central slice of prediction

            for contrast in modalities:
                if format == 'nii.gz':
                    full_img = NII.load(bids_val_ds.bids_list[idx][contrast], seg=False)
                    img_array = full_img.get_array()
                elif format == 'fnio':
                    img_array = fnio.load(str(bids_val_ds.bids_list[idx][contrast]))

                img_slice = get_central_slice(img_array, axis)
                plot_slices(img_slice, pred_slice,plt_title='Prediction '+ suffix , save_path= save_path / f"slices" /  f"{subject}-{contrast}-{conf.axis}-slice-pred-{suffix}.png", show=False)
                # if save_gts:
                #     plot_slices(img_slice, gt_slice, plt_title='Ground Truth',save_path=save_path / f"slices" / f"{subject}-{contrast}-{conf.axis}-slice-gt.png", show = False)
                plot_slices(img_slice, gt_slice, plt_title='Ground Truth',save_path=save_path / f"slices" / f"{subject}-{contrast}-{conf.axis}-slice-gt.png", show = False)

        
        if conf.soft:
            soft_save_path = save_dir / model_name / subject / f"channelwise"
            soft_slice_path : Path = soft_save_path / f"slices"

            if not soft_slice_path.exists():
                soft_slice_path.mkdir(parents=True)

            # load soft ground truth segmentation
            soft_gt = dicts['soft_seg'] 

            with torch.no_grad():
                logits = model(img_tensor)  # get logits from model

                if conf.activation == 'softmax':
                    probs = softmax(logits) # apply softmax to get probabilities
                elif conf.activation == 'relu':
                    if bool(relu(logits).max()): # checking if the max value of the relu is not zero
                        probs = relu(logits)/relu(logits).max()
                    else: 
                        probs = relu(logits)

                probs_cpu = probs.cpu() # move tensor to cpu
                del logits, probs # delete tensors to free up memory

            prob_channels = []
            gt_channels = []

            soft_gt = soft_gt.unsqueeze(0) # add batch dimension

            for channel in range(probs_cpu.shape[1]):
                prob_channels.append(pad(probs_cpu[:, channel])) # pad probs_cpu to original image size
                gt_channels.append(pad(soft_gt[:, channel]))     # pad soft_gt to original image size

            probs_padded = torch.stack(prob_channels, dim=1) # stack channels to get shape (1, C, H, W, D)
            gt_padded = torch.stack(gt_channels, dim=1) # stack channels to get shape (C, H, W, D)
            
            probs_padded = probs_padded.squeeze(0) # get rid of batch dimension
            gt_padded = gt_padded.squeeze(0) # get rid of batch dimension
            
            probs_array = probs_padded.numpy() # convert to numpy array (dtype: float32)
            gt_array = gt_padded.numpy() # convert to numpy array (dtype: float32)

            if conf.round:
                probs_array = probs_array.round(conf.round) # round probabilities to given number of decimals
                gt_array = gt_array.round(conf.round) # round probabilities to given number of decimals
            
            # load MRI as NII object
            sample_img_path = base_path / f"{subject}-t1c.nii.gz"
            nii_img = NII.load(sample_img_path,seg=False)
            nii_img.set_dtype_(np.float32)

            if format == 'nii.gz':
                full_img = NII.load(bids_val_ds.bids_list[idx]['t1c'], seg=False)
                img_array = full_img.get_array()
            elif format == 'fnio':
                img_array = fnio.load(str(bids_val_ds.bids_list[idx]['t1c']))

            for channel in range(probs_array.shape[0]):
                nii_prob: NII = nii_img.set_array(probs_array[channel])  # get prediction as nii object
                nii_soft_gt: NII = nii_img.set_array(gt_array[channel])  # get prediction as nii object

                if save_niftis:
                    nii_prob.save(soft_save_path / f"niftis" / f"{subject}-prob-class{channel}-{suffix}.nii.gz")    # save prediction as nifti file to view it in ITK Snap
                    nii_soft_gt.save(soft_save_path / f"niftis" / f"{subject}-soft_gt-class{channel}-sigma-{train_opt.sigma}.nii.gz")

                if save_slices:
                    prob_slice = get_central_slice(probs_array[channel], axis) # get central slice of predicted probabilities
                    gt_slice = get_central_slice(gt_array[channel], axis)      # get central slice of soft ground truth probabilities

                    img_slice = get_central_slice(img_array, axis)
                
                    plot_slices(img_slice, prob_slice,plt_title=f"Predicted probability channel {channel} {suffix} ", save_path=soft_save_path / f"slices" / f"{subject}-t1c-{conf.axis}-slice-probability-class-{channel}-{suffix}.png",omit_background=True, show=False)
                    plot_slices(img_slice, gt_slice, plt_title=f"Soft GT probability channel {channel} sigma {train_opt.sigma}", save_path=soft_save_path / f"slices" / f"{subject}-t1c-{conf.axis}-slice-soft_gt-class-{channel}-sigma-{train_opt.sigma}.png", show=False)
 
        if samples > 0:
            if idx == (samples - 1):
                break 

        if test_loop:     
            # for testing purposes -> exit after one iteration
            break
