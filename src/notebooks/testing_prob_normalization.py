# %%
import sys
import os
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
sys.path.append(str(file.parents[1]))
# %%
from argparse import Namespace
from data.bids_dataset import BidsDataset
from utils.brats_tools import temperature_scaled_softmax
import torch
from utils.brats_tools import plot_slices, get_central_slice
relu = torch.nn.ReLU()
softmax = torch.nn.Softmax(dim=0)

data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge'
# %%
def print_unique_etc(tensor):
    print(f"unique values in soft gt: {torch.unique(tensor)}")
    print(f"unique sums across all channels {torch.unique(tensor.sum(dim=0), return_counts=True)}") 

def l1_norm(probs):
    probs = probs / probs.sum(dim=0, keepdim=True)  # for tensor of shape (C, H, W, D), so without batch dim
    return probs

def exp_norm(probs, scaling_factor = 0.1):
    exp_tensor = torch.exp(scaling_factor * probs)
    probs = exp_tensor / exp_tensor.sum(dim = 0, keepdim = True)    # for tensor of shape (C, H, W, D), so without batch dim
    return probs

def prob_normalization(probs, temperature = 0.1, scale  = 0.1):
    #print(f"Soft Probs:")
    #print_unique_etc(probs)
    print(f"---------------")
    print(f"Temperature Softmaxed Probs")
    print_unique_etc(temperature_scaled_softmax(probs, temperature=temperature))
    print(f"---------------")
    print(f"L1 Normalized Probs")
    print_unique_etc(l1_norm(probs))

def test_temp_softmax(probs):
    values = (0.1, 0.15, 0.2)
    #print(f"normal vals:")
    #print_unique_etc(probs)
    print(f"------------------------------------")
    for temp in values:
        print(f"Temperature: {temp}")
        print_unique_etc(temperature_scaled_softmax(probs, temperature=temp))
        print(f"------------------------------------")

# %%
dilate = 1
sigma = 0.5
temp = None

conf = Namespace(resize = (152, 192, 144), contrast = 'multimodal', one_hot = False,  sigma = sigma, do2D = False, soft=False, dilate=dilate, binary=False, ds_factor=None, experiment = 1, softmax_temperature=temp)
bids_ds = BidsDataset(conf, data_dir+'/train')

sample = bids_ds[0]
sample_img = sample['img']
sample_gt = sample['seg']
sample_soft_gt = sample['soft_seg']
# %% 
#print_unique_etc(sample_soft_gt)
# print_unique_etc(sample_soft_gt.round(decimals = 6))
# print_unique_etc(sample_soft_gt.round(decimals = 5))
print_unique_etc(sample_soft_gt.round(decimals = 4))
print_unique_etc(sample_soft_gt.round(decimals = 3))
# %%
dec = 3
print(f"L1 Normalized Probs")
print_unique_etc(l1_norm(sample_soft_gt).round(decimals = dec))
print(f"Temperature Softmaxed Probs w Temp 0.1")
print_unique_etc(temperature_scaled_softmax(sample_soft_gt, temperature=0.1).round(decimals = dec))
# %%

# %%
import numpy as np
from TPTBox import NII

## SANITY CHECK
#  %%
dilate = 0
sigma = 0.5
dec = 4
temp = None # not applied through BidsDataset but directly inside the loop

conf = Namespace(resize = (152, 192, 144), contrast = 'multimodal', one_hot = False,  sigma = sigma, do2D = False, soft=False, dilate=dilate, binary=False, ds_factor=None, experiment = 1, softmax_temperature=None)
bids_ds = BidsDataset(conf, data_dir+'/train')

sample = bids_ds[0]
sample_img = sample['img']
sample_gt = sample['seg']
sample_soft_gt = sample['soft_seg']
# %%
# # load sample GT segmentation as NII object
gt_path = bids_ds.bids_list[0]['seg']
gt = NII.load(gt_path.with_suffix('.nii.gz'), seg=True)

# %%
print(f"Experiment without normalization and rounding and dilate = {dilate}, sigma= {sigma}, dec = {dec}")
#print(f"L1 Norm Experiment with dilate = {dilate}, sigma= {sigma}, dec = {dec}")
#print(f"temp Scaled Softmax Experiment with temp = {0.1}, dilate = {dilate}, sigma= {sigma}, dec = {dec}")
#print(f"Rounding with temp = {temp}, dilate = {dilate}, sigma= {sigma}, dec = {dec} ")
fn = 0
tp = 0
fp = 0
wrong = 0

start_idx = 0
num_range = 10
for idx in range(num_range):
    sample = bids_ds[start_idx + idx]
    sample_gt = sample['seg']
    sample_soft_gt = sample['soft_seg']

    ## PROCESSING HERE
    #sample_soft_gt = l1_norm(sample_soft_gt).round(decimals = dec)
    #sample_soft_gt = temperature_scaled_softmax(sample_soft_gt, temperature=0.1).round(decimals = dec)
    #sample_soft_gt = sample_soft_gt.round(decimals = dec)
    ##

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