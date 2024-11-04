# %%
import sys
import os
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
sys.path.append(str(file.parents[1]))
# %%
from TPTBox import NII
from argparse import Namespace 
import torch
import numpy as np
from data.bids_dataset import BidsDataset
# %%

data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge'
dilate = 0
conf = Namespace(resize = (152, 192, 144), contrast = 'multimodal', one_hot = False, sigma = 0.5, do2D = False, soft=False, dilate=dilate, binary=False, ds_factor=None, experiment = 1)
bids_ds = BidsDataset(conf, data_dir+'/train')

# %%
# load sample GT segmentation as NII object
gt_path = bids_ds.bids_list[0]['seg']
gt = NII.load(gt_path.with_suffix('.nii.gz'), seg=True)
# %%
fn = 0
tp = 0
fp = 0
wrong = 0

start_idx = 0
num_range = 50
for idx in range(num_range):
    sample = bids_ds[start_idx + idx]
    sample_gt = sample['seg']
    sample_soft_gt = sample['soft_seg']

    gt : NII = gt.set_array(sample_gt.squeeze(0).numpy())

    rebinarized_gt = torch.argmax(sample_soft_gt, dim = 0).numpy()
    rebinarized_nii: NII = gt.set_array(rebinarized_gt)  # get prediction as nii object

    difference_nifti = NII.get_segmentation_difference_to(rebinarized_nii, gt, ignore_background_tp=True)
    difference_array = difference_nifti.get_array()
    values, counts = np.unique(difference_array, return_counts=True)
    print(f"{start_idx + idx}: Unique Values {values} and counts: {counts}")
    ## (0: BG, 1: FN, 2: TP, 3: FP, 4: Wrong label)")
    for i in range(len(values)):
        if values[i] == 1:
            fn = fn + counts[i]
        elif values[i] == 2:
            tp = tp + counts[i]
        elif values[i] == 3:
            fp = fp + counts[i]
        elif values[i] == 4:
            wrong = wrong + counts[i]

print(f" For {num_range} samples, starting at {start_idx}, there were:")
print(f"FN: {fn}, on avg: {fn/num_range}")
print(f"TP: {tp}, on avg: {tp/num_range}")
print(f"FP: {fp}, on avg: {fp/num_range}")
print(f"Wrong: {wrong}, on avg: {wrong/num_range}")
# %%