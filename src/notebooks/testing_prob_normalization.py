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
    values = (0.05, 0.1, 0.15)
    #print(f"normal vals:")
    #print_unique_etc(probs)
    print(f"------------------------------------")
    for temp in values:
        print(f"Temperature: {temp}")
        print_unique_etc(temperature_scaled_softmax(probs, temperature=temp))
        print(f"------------------------------------")

# %%
dilate = 0
sigma = 0.4
temp = 0.1
soft_gt_norm = 'temp_scaled_softmax'
round = 3

conf = Namespace(resize = (152, 192, 144), contrast = 'multimodal', one_hot = False,  sigma = sigma, do2D = False, soft=False, dilate=dilate, binary=False, ds_factor=None, experiment = 1, softmax_temperature=temp, soft_gt_norm = soft_gt_norm, round = round )
bids_ds = BidsDataset(conf, data_dir+'/train')

sample = bids_ds[0]
sample_img = sample['img']
sample_gt = sample['seg']
sample_soft_gt = sample['soft_seg']
# %% 
print(f"dilate={dilate}, sigma={sigma}, temp={temp}, soft_gt_norm={soft_gt_norm}, round={round} ")
print_unique_etc(sample_soft_gt)
# print_unique_etc(sample_soft_gt.round(decimals = 6))
# print_unique_etc(sample_soft_gt.round(decimals = 5))
# print_unique_etc(sample_soft_gt.round(decimals = 4))
# print_unique_etc(sample_soft_gt.round(decimals = 3))
# %%
# dec = 3
# print(f"L1 Normalized Probs")
# print_unique_etc(l1_norm(sample_soft_gt).round(decimals = dec))
# print(f"Temperature Softmaxed Probs w Temp 0.1")
# print_unique_etc(temperature_scaled_softmax(sample_soft_gt, temperature=0.1).round(decimals = dec))
# %%