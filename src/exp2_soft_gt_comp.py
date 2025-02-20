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
from utils.brats_tools import get_central_slice, plot_slices, postprocessing
#from monai.transforms import ResizeWithPadOrCrop, CastToType
from torch import nn
from data.bids_dataset import modalities, create_bids_path_list_of_dicts, BidsDataset, BidsDataModule
import lightning.pytorch as pl
import time
import torchmetrics.functional as mF
import pandas as pd

# file = Path(__file__).resolve()
# sys.path.append(str(file.parents[1]))
# sys.path.append(str(file.parents[2]))

def compute_metrics_summary(file_path, save_dir, suffix):
    # Load the TSV file
    data = pd.read_csv(file_path, sep='\t')

    # Compute average and standard deviation for each metric (excluding 'subject_name')
    metrics_summary = {}
    metrics = data.columns.drop('subject_name')  # Exclude the 'subject_name' column

    for metric in metrics:
        avg = data[metric].mean()
        std = data[metric].std()
        metrics_summary[metric] = (avg, std)

    # Transform the data into a format suitable for CSV
    rows = []
    for metric, (avg, std) in metrics_summary.items():
        rows.append({
            'Metric': metric,
            'Average': avg,
            'StdDev': std
        })

    # Create a DataFrame for saving
    summary_df = pd.DataFrame(rows)

    # Save the DataFrame as a CSV file
    summary_df.to_csv(save_dir.joinpath(f"{suffix}_summary.csv"), index=False)

    # Display the DataFrame
    print(summary_df)

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
    
    parser.add_argument("-setup_dir", type=str, default = None, help="Path to a model log folder using the soft GTs you want to evaluate. Script will automatically obtain the path of the best cpkt (best DiceFG)")
    parser.add_argument("-data_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test", help="Path to the data directory")
    parser.add_argument("-eval_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval", help="Path to the directory where the eval.tsv should be saved")
    parser.add_argument("-ref_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs/brats/exp_2/3D_UNet/3D_UNet_v1_lr0.0001_ce_1.0_hard_1.0_soft_dice_1.0_down_factor_2_sigma_0.5_binary_softmax_exp2_and_3_baseline", help="Path to model log folder of reference soft GT and hard GT")
    
    parser.add_argument("-suffix", type=str, default = None, help="Suffix for saved predictions")

    parser.add_argument("-postprocessing", action="store_true", help = "In case you don't want to postprocess the outputs of the models, by rounding to 3 decimals and smoothing values below 0.05 and above 0.95.")
    parser.add_argument("-smoothing", action="store_true", help = "In case you want to postprocess the output of the model, but not smooth values below 0.05 and above 0.95.")

    parser.add_argument("-postprocess_gt", action="store_true", help = "In case you want to apply the same postprocessing, to the model outputs and soft ground truths")

    return parser

if __name__ == '__main__':

    torch.set_grad_enabled(False)

    t = time.process_time()
    print(f"Start Time: {t} seconds")

    parser = parse_inf_param()
    conf = parser.parse_args()

    eval_dir : Path = Path(conf.eval_dir)
    data_dir : Path = Path(conf.data_dir)
    setup_path : Path = Path(conf.setup_dir)
    ref_path : Path = Path(conf.ref_dir)
    suffix = conf.suffix

    # data_dir : Path = Path("data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test")
    # eval_dir : Path = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval")
    # setup_path : Path = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs/brats/exp_2/3D_UNet/3D_UNet_v4_lr0.0001_soft_1.0_mse_1.0_down_factor_2_sigma_0.125_binary_linear_exp_3_mse_sigma_0.125")
    #ref_path :Path = ("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs/brats/exp_2/3D_UNet/3D_UNet_v1_lr0.0001_ce_1.0_hard_1.0_soft_dice_1.0_down_factor_2_sigma_0.5_binary_softmax_exp2_and_3_baseline")
    # suffix = "test"

    comp_ckpt_path = get_best_checkpoint(setup_path)
    ref_ckpt_path = get_best_checkpoint(ref_path)


    print(f"Best checkpoint: {comp_ckpt_path.name}")
        
    # try if torch.load works, otherwise raise an error and exit
    try:
        checkpoint = torch.load(comp_ckpt_path)
    except Exception as e:
        print(f"Error: Invalid checkpoint path or file not found: {comp_ckpt_path}")
        sys.exit(1)

    # load hparams from checkpoint to get contrast
    hparams_comp = LitUNetModule.load_from_checkpoint(comp_ckpt_path).hparams
    comp_train_opt = hparams_comp.get('opt')

    hparams_ref = LitUNetModule.load_from_checkpoint(ref_ckpt_path).hparams
    ref_train_opt = hparams_ref.get('opt')

    ###### CREATING TSV FOR SAVING THE SOFT METRICS
    output_file = os.path.join(eval_dir, f"{suffix}_comparison.tsv")
    
    # Define the column names
    columns = [
    "subject_name", 
    "mse_soft", "mae_soft",
    "mse_soft_hard", "mae_soft_hard", 
    "avd_soft", "rvd_soft", 
    "avd_soft_hard", "rvd_soft_hard", 
    ]


    # create BidsDataset instance
    bids_test_ref_ds = BidsDataset(ref_train_opt, data_dir)
    bids_test_comp_ds = BidsDataset(comp_train_opt, data_dir)

    print(f"Reference Soft GT Experiment: {ref_train_opt.experiment}")
    if ref_train_opt.experiment == 3:
        print(f"Sigma of Reference Soft GT: {ref_train_opt.sigma}")
    print(f"Soft GT to be compared: {comp_train_opt.experiment}")
    print(f"Sigma of compared soft GT: {comp_train_opt.sigma}")

    # Open the .tsv file for writing
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter='\t')
        writer.writeheader()  # Write the header row

        for idx, dict in enumerate(zip(bids_test_ref_ds,bids_test_comp_ds)):
            down_dict = dict[0]
            mod_dict = dict[1]

            ref_subject = bids_test_ref_ds.bids_list[idx]['subject']
            mod_subject = bids_test_comp_ds.bids_list[idx]['subject']

            assert ref_subject == mod_subject, f"ref_subject: {ref_subject} should be equal to mod_subject: {mod_subject}"

            print(f"---------------------------------")
            print(f"Subject: {ref_subject}")

            ref_gt = down_dict['seg']       # shape [1, 80, 96, 72]
            ref_soft_gt = down_dict['soft_seg'] #shape [2, 80, 96, 72]

            mod_soft_gt = mod_dict['soft_seg']

            # ### PRINT IMAGES FOR TESTING ####

            # down_img = down_dict['img'][0]
            # down_img_slice = get_central_slice(down_img)

            # down_soft_gt_slice = get_central_slice(ref_soft_gt[1])
            # mod_soft_gt_slice = get_central_slice(mod_soft_gt[1])

            # plot_slices(down_img_slice, mod_soft_gt_slice, show = True, save_path=f"{eval_dir}/{suffix}_img.png", gt_slice=down_soft_gt_slice)

            # ### PRINT IMAGES FOR TESTING ####


            if conf.postprocessing:
                mod_soft_gt = postprocessing(mod_soft_gt, 3, conf.smoothing)    # kills all values < 0, rounds to 3 decimals and smoothes values below 0 and above 0.95

            if conf.postprocess_gt:
                ref_soft_gt = postprocessing(ref_soft_gt, 3, conf.smoothing)    # kills all values < 0, rounds to 3 decimals and smoothes values below 0 and above 0.95

            ref_gt_vol = ref_gt.sum()     # just sum all ones
            ref_soft_gt_vol = ref_soft_gt[1].sum() # sum all float values along the foreground dimension (channel 1)

            mod_soft_gt_vol = mod_soft_gt[1].sum() # sum all float values along the foreground dimension (channel 1)

            ## EVAL OF SOFT SCORES/PROBABILITES OUTPUT BY MODEL AND NATURALLY CREATED SOFT GTs THROUGH DOWNSAMPLING
            mse_soft = mF.mean_squared_error(mod_soft_gt[1], ref_soft_gt[1])
            mae_soft = mF.mean_absolute_error(mod_soft_gt[1], ref_soft_gt[1])

            mse_soft_hard = mF.mean_squared_error(mod_soft_gt[1], ref_gt.squeeze(0))
            mae_soft_hard = mF.mean_absolute_error(mod_soft_gt[1], ref_gt.squeeze(0))

            avd_soft, rvd_soft = avd_rvd(mod_soft_gt_vol, ref_soft_gt_vol)
            avd_soft_hard, rvd_soft_hard = avd_rvd(mod_soft_gt_vol, ref_gt_vol)   # difference between gaussian created soft gt and hard gt volume

            data = {
                "subject_name": ref_subject,
                "mse_soft": mse_soft.item(),
                "mae_soft": mae_soft.item(),
                "mse_soft_hard": mse_soft_hard.item(),
                "mae_soft_hard": mae_soft_hard.item(),
                "avd_soft": avd_soft.item(),
                "rvd_soft": rvd_soft.item(),
                "avd_soft_hard": avd_soft_hard.item(),
                "rvd_soft_hard": rvd_soft_hard.item(),
            }

            # Write the row for the current subject
            writer.writerow(data)

    compute_metrics_summary(output_file, eval_dir, suffix)

    elapsed_time = time.process_time() - t
    print(f"Process took this much time: {elapsed_time} seconds")