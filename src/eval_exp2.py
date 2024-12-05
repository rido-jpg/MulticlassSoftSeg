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

def _get_gt_paths(gt_dir : Path) -> list:
    gt_dir : Path = Path(gt_dir)
    gt_paths = list(gt_dir.rglob("*seg.nii.gz"))
    return gt_paths

def _extract_brats_id(filename):
    # Define the regular expression pattern to match BraTS-GLI-XXXXXX-XXX
    pattern = r"(BraTS-GLI-\d{5}-\d{3})"
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)  # Return the matched part
    else:
        return None  # Return None if no match is found
    
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
    
    parser.add_argument("-pred_dir", type=str, default = None, help="Path to a bids type folder structure containing the soft predictions.")
    #parser.add_argument("-data_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test", help="Path to the data directory")
    parser.add_argument("-eval_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval", help="Path to the directory where the eval.tsv should be saved")
    
    parser.add_argument("-suffix", type=str, default = None, help="Suffix for saved predictions")

    parser.add_argument("-postprocessing", action="store_true", help = "In case you don't want to postprocess the outputs of the models, by rounding to 3 decimals and smoothing values below 0.05 and above 0.95.")
    parser.add_argument("-smoothing", action="store_true", help = "In case you want to postprocess the output of the model, but not smooth values below 0.05 and above 0.95.")

    parser.add_argument("-binarize_soft_gt", action="store_true", help = "compares binarized_soft_gt")

    return parser

if __name__ == '__main__':

    torch.set_grad_enabled(False)

    t = time.process_time()
    print(f"Start Time: {t} seconds")

    parser = parse_inf_param()
    conf = parser.parse_args()

    eval_dir : Path = Path(conf.eval_dir)
    pred_path : Path = Path(conf.pred_dir)

    suffix = conf.suffix

    og_hard_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test")
    scaling_factor = 2*2*2

    og_hard_gt_list = _get_gt_paths(og_hard_dir)

    soft_ds_gt_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test_ds")
    soft_ds_gt_list = _get_gt_paths(soft_ds_gt_dir)

    preds_soft_list = _get_gt_paths(pred_path)

    ###### CREATING TSV FOR SAVING THE SOFT METRICS
    output_file = os.path.join(eval_dir, f"{suffix}_comparison.tsv")

    # Define the column names
    columns = [
    "subject_name", 
    "mse_soft", "mae_soft",
    "fmse_soft", "fmae_soft",
    "avd_og_hard", "rvd_og_hard", 
    ]
    # Open the .tsv file for writing
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter='\t')
        writer.writeheader()  # Write the header row

        for idx, entry in enumerate(zip(og_hard_gt_list,soft_ds_gt_list,preds_soft_list)):
            og_hard_gt_path = entry[0]
            soft_gt_path = entry[1]
            pred_soft_path = entry[2]
            
            og_subject = _extract_brats_id(str(og_hard_gt_path))
            soft_subject = _extract_brats_id(str(soft_gt_path))
            preds_soft_subject = _extract_brats_id(str(pred_soft_path))

            assert og_subject == preds_soft_subject == soft_subject, f"og_subject: {og_subject} should be equal to soft_subject: {soft_subject} and mod_subject: {mod_subject}"

            print(f"---------------------------------")
            print(f"Subject: {og_subject}")

            og_hard_gt_arr = NII.load(og_hard_gt_path, True).get_array()       # shape [1, 80, 96, 72]
            og_hard_gt = torch.from_numpy(og_hard_gt_arr)

            ds_soft_gt_arr = NII.load(soft_gt_path, False).get_array()
            ds_soft_gt = torch.from_numpy(ds_soft_gt_arr) #shape [2, 80, 96, 72]
            
            pred_soft_arr = NII.load(pred_soft_path, False).get_array
            pred_soft = torch.from_numpy(pred_soft_arr)

            # ### PRINT IMAGES FOR TESTING ####

            # down_img = down_dict['img'][0]
            # down_img_slice = get_central_slice(down_img)

            # down_soft_gt_slice = get_central_slice(ref_soft_gt[1])
            # mod_soft_gt_slice = get_central_slice(mod_soft_gt[1])

            # plot_slices(down_img_slice, mod_soft_gt_slice, show = True, save_path=f"{eval_dir}/{suffix}_img.png", gt_slice=down_soft_gt_slice)

            # ### PRINT IMAGES FOR TESTING ####

            if conf.postprocessing:
                pred_soft = postprocessing(pred_soft, 3, conf.smoothing)    # kills all values < 0, rounds to 3 decimals and smoothes values below 0 and above 0.95

            og_hard_gt_vol = og_hard_gt.count_nonzero()       # count all non-zero values (as there are 1,2 and 3s)

            ds_soft_gt_vol = ds_soft_gt.sum()                  # sum all float values (only foreground channel loaded)

            pred_soft_vol = pred_soft.sum()     # sum all float values (only foreground channel loaded)

            avd_og_hard, rvd_og_hard = avd_rvd(pred_soft_vol * scaling_factor, og_hard_gt_vol)   # vol difference between soft prediction and original hard gt volume

            mse_soft = mF.mean_squared_error(pred_soft, ds_soft_gt)
            mae_soft = mF.mean_absolute_error(pred_soft, ds_soft_gt)

            img_area = pred_soft.view(-1).shape[0]
            fore_mask = (pred_soft > 0) & (ds_soft_gt > 0)
            fore_area = torch.count_nonzero(fore_mask)
            fmse_soft = mse_soft*img_area/fore_area
            fmae_soft = mae_soft*img_area/fore_area

            data = {
                "subject_name": og_subject,
                "mse_soft": mse_soft.item(),
                "mae_soft": mae_soft.item(),
                "fmse_soft": fmse_soft.item(),
                "fmae_soft": fmae_soft.item(),
                "avd_og_hard": avd_og_hard.item(),
                "rvd_og_hard": rvd_og_hard.item(),
            }

            # Write the row for the current subject
            writer.writerow(data)

    compute_metrics_summary(output_file, eval_dir, suffix)

    elapsed_time = time.process_time() - t
    print(f"Process took this much time: {elapsed_time} seconds")