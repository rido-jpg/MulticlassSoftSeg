# %%
import os
import torch
import re
import csv
from pathlib import Path
from TPTBox import NII
#from utils.brats_tools import get_central_slice, plot_slices, postprocessing
import time
import pandas as pd

# file = Path(__file__).resolve()
# sys.path.append(str(file.parents[1]))
# sys.path.append(str(file.parents[2]))
# %%
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
 # %%
if __name__ == '__main__':

    t = time.process_time()
    print(f"Start Time: {t} seconds")

    eval_dir : Path = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval")
    data_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test")
    
    suffix = "ds_soft_rebin_hard_og_hard_vol"
    #suffix = "test"
    scaling_factor = 2*2*2

    og_hard_gt_list = _get_gt_paths(data_dir)

    soft_ds_gt_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test_ds")
    soft_ds_gt_list = _get_gt_paths(soft_ds_gt_dir)


    ###### CREATING TSV FOR SAVING THE SOFT METRICS
    output_file = os.path.join(eval_dir, f"{suffix}_comparison.tsv")

    # Define the column names
    columns = [
    "subject_name", 
    "avd_ds_soft_og_hard", "rvd_ds_soft_og_hard",
    "avd_rebin_hard_og_hard", "rvd_rebin_hard_og_hard"
    ]
    # %%
    # Open the .tsv file for writing
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns, delimiter='\t')
        writer.writeheader()  # Write the header row

        for idx, dict in enumerate(zip(og_hard_gt_list,soft_ds_gt_list)):
            og_hard_gt_path = dict[0]
            soft_gt_path = dict[1]
            
            og_subject = _extract_brats_id(str(og_hard_gt_path))
            soft_subject = _extract_brats_id(str(soft_gt_path))

            assert og_subject == soft_subject, f"og_subject: {og_subject} should be equal to soft_subject: {soft_subject}"

            print(f"---------------------------------")
            print(f"Subject: {og_subject}")

            og_hard_gt_arr = NII.load(og_hard_gt_path, True).get_array()
            og_hard_gt = torch.from_numpy(og_hard_gt_arr).type(torch.uint8)   # [240, 240, 155]
            ds_soft_gt_arr = NII.load(soft_gt_path, False).get_array()
            ds_soft_gt = torch.from_numpy(ds_soft_gt_arr) #   [120, 120, 78]

            rebin_hard_gt = (ds_soft_gt >= 0.5).type(torch.uint8)   # größer gleich, da torch.round bei 0.5 abrundet
            # %%
            # print(f"og_hard_gt.shape {og_hard_gt.shape}")
            # print(f"ds_soft_gt.shape {ds_soft_gt.shape}")
            # print(f"rebin_hard_gt.shape {rebin_hard_gt.shape}")
            # # %%
            # print(f"torch.unique(og_hard_gt, return_counts=True) {torch.unique(og_hard_gt, return_counts=True)}")
            # print(f"torch.unique(ds_soft_gt.shape, return_counts=True) {torch.unique(ds_soft_gt, return_counts=True)}")
            # print(f"torch.unique(rebin_hard_gt, return_counts=True) {torch.unique(rebin_hard_gt, return_counts=True)}")
        
            # ### PRINT IMAGES FOR TESTING ####

            # down_img = down_dict['img'][0]
            # down_img_slice = get_central_slice(down_img)

            # down_soft_gt_slice = get_central_slice(ref_soft_gt[1])
            # mod_soft_gt_slice = get_central_slice(mod_soft_gt[1])

            # plot_slices(down_img_slice, mod_soft_gt_slice, show = True, save_path=f"{eval_dir}/{suffix}_img.png", gt_slice=down_soft_gt_slice)

            # ### PRINT IMAGES FOR TESTING ####
            # %%
            og_hard_gt_vol = og_hard_gt.count_nonzero()       # count all non-zero values (as there are 1,2 and 3s)
            ds_soft_gt_vol = ds_soft_gt.sum()                  # sum all float values (only foreground channel loaded)
            rebin_hard_gt_vol = rebin_hard_gt.count_nonzero() 

            # print(f"og_hard_gt_vol {og_hard_gt_vol}, downscaled:{og_hard_gt_vol/scaling_factor} ")
            # print(f"ds_soft_gt_vol {ds_soft_gt_vol}, upscaled: {ds_soft_gt_vol*scaling_factor}")
            # print(f"rebin_hard_gt_vol {rebin_hard_gt_vol}, upscaled: {rebin_hard_gt_vol*scaling_factor}")

            # %%
            avd_ds_soft_og_hard, rvd_ds_soft_og_hard = avd_rvd(ds_soft_gt_vol*scaling_factor, og_hard_gt_vol)
            avd_rebin_hard_og_hard, rvd_rebin_hard_og_hard = avd_rvd(rebin_hard_gt_vol*scaling_factor, og_hard_gt_vol)
        
            
            # %%
            data = {
            "subject_name": og_subject,
            "avd_ds_soft_og_hard": avd_ds_soft_og_hard.item(),
            "rvd_ds_soft_og_hard": rvd_ds_soft_og_hard.item(),
            "avd_rebin_hard_og_hard": avd_rebin_hard_og_hard.item(),
            "rvd_rebin_hard_og_hard": rvd_rebin_hard_og_hard.item(),
            }

            # Write the row for the current subject
            writer.writerow(data)


    compute_metrics_summary(output_file, eval_dir, suffix)

    elapsed_time = time.process_time() - t
    print(f"Process took this much time: {elapsed_time} seconds")