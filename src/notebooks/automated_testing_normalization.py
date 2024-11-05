import sys
import os
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
sys.path.append(str(file.parents[1]))

from argparse import Namespace
from data.bids_dataset import BidsDataset
from utils.brats_tools import temperature_scaled_softmax, l1_norm
import itertools
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import time
from TPTBox import NII

data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge'

# Define the parameters for all combinations
dilate_values = [0, 1]
sigma_values = [0.125, 0.2, 0.25, 0.3, 0.4, 0.5]
dec_values = [3, 4, 5, 6]
temp_values = [0.05, 0.1, 0.15, 0.2]  # None is ignored

# Directory for saving results
output_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/notebooks/sanity_check_results")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize list to collect summary statistics for CSV
summary_stats = []

# Load the NII skeleton once at the beginning
conf = Namespace(
    resize=(152, 192, 144), contrast='multimodal', one_hot=False,
    sigma=sigma_values[0], do2D=False, soft=False, dilate=dilate_values[0],
    binary=False, ds_factor=None, experiment=1
)
bids_ds_temp = BidsDataset(conf, data_dir+'/train')
gt_path = bids_ds_temp.bids_list[0]['seg']
gt_skeleton = NII.load(gt_path.with_suffix('.nii.gz'), seg=True)

# Only create BidsDataset for each (dilate, sigma) combination
for dilate, sigma in itertools.product(dilate_values, sigma_values):
    print(f"Creating dataset for dilate={dilate}, sigma={sigma}")
    conf = Namespace(
        resize=(152, 192, 144), contrast='multimodal', one_hot=False,
        sigma=sigma, do2D=False, soft=False, dilate=dilate,
        binary=False, ds_factor=None, experiment=1
    )
    bids_ds = BidsDataset(conf, data_dir+'/train')

    # For each unique dataset, test all combinations of dec, temp, and methods
    for dec, temp in itertools.product(dec_values, temp_values):
        processing_methods = [
            ("temperature_scaled_softmax", lambda x: temperature_scaled_softmax(x, dim= 0, temperature=temp).round(decimals = dec)),
            ("l1_norm", lambda x: l1_norm(x).round(decimals = dec)),
            ("round_only", lambda x: x.round(decimals = dec))
        ]

        for method_name, processing_func in processing_methods:
            fn, tp, fp, wrong = 0, 0, 0, 0  # Reset metrics for each method
            num_samples = 50  # Adjust as necessary
            start_idx = 0

            # Track time for the method
            start_time = time.time()

            # Initialize list to collect sample-level statistics for CSV/TSV
            sample_stats = []

            for idx in range(num_samples):
                sample = bids_ds[start_idx + idx]
                sample_gt = sample['seg']
                sample_soft_gt = sample['soft_seg']
                
                # Apply the chosen processing method
                processed_soft_gt = processing_func(sample_soft_gt)

                # Binarize the processed soft GT
                rebinarized_gt = torch.argmax(processed_soft_gt, dim=0).numpy()
                
                # Set the ground truth and prediction using the skeleton
                gt_instance = gt_skeleton.set_array(sample_gt.squeeze(0).numpy())
                rebinarized_nii = gt_skeleton.set_array(rebinarized_gt)  # Prediction as NII object
                
                # Compare modified GT to the original GT
                difference_nifti = NII.get_segmentation_difference_to(rebinarized_nii, gt_instance, ignore_background_tp=True)
                difference_array = difference_nifti.get_array()
                values, counts = np.unique(difference_array, return_counts=True)
                
                sample_fn, sample_tp, sample_fp, sample_wrong = 0, 0, 0, 0
                for val, count in zip(values, counts):
                    if val == 1:
                        sample_fn += count
                    elif val == 2:
                        sample_tp += count
                    elif val == 3:
                        sample_fp += count
                    elif val == 4:
                        sample_wrong += count
                
                # Accumulate totals
                fn += sample_fn
                tp += sample_tp
                fp += sample_fp
                wrong += sample_wrong

                # Append sample-level stats to list
                sample_stats.append({
                    "sample_index": start_idx + idx,
                    "FN": sample_fn,
                    "TP": sample_tp,
                    "FP": sample_fp,
                    "Wrong": sample_wrong
                })

            # Save sample-level statistics to CSV for this configuration and method
            sample_stats_df = pd.DataFrame(sample_stats)
            sample_stats_file = output_dir / f"sample_stats_dilate_{dilate}_sigma_{sigma}_dec_{dec}_temp_{temp}_{method_name}.csv"
            sample_stats_df.to_csv(sample_stats_file, index=False)

            # Calculate averages
            avg_fn = fn / num_samples
            avg_tp = tp / num_samples
            avg_fp = fp / num_samples
            avg_wrong = wrong / num_samples

            # Record end time and calculate duration
            end_time = time.time()
            duration = end_time - start_time

            # Append configuration statistics to summary
            summary_stats.append({
                "dilate": dilate,
                "sigma": sigma,
                "dec": dec,
                "temp": temp,
                "method": method_name,
                "avg_fn": avg_fn,
                "avg_tp": avg_tp,
                "avg_fp": avg_fp,
                "avg_wrong": avg_wrong,
                "duration_sec": duration
            })

            print(f"Completed configuration: dilate={dilate}, sigma={sigma}, dec={dec}, temp={temp}, method={method_name}")

# Save summary statistics to CSV
summary_df = pd.DataFrame(summary_stats)
summary_df.to_csv(output_dir / "summary_statistics.csv", index=False)