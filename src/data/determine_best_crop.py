'''
Script that calculates and determines the best crop for a BIDS style dataset
'''

import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from TPTBox import NII
import matplotlib.pyplot as plt
import numpy as np
from utils.brats_tools import get_central_slice, plot_slices
from data.bids_dataset import create_bids_path_list_of_dicts, modalities
import utils.fastnumpyio.fastnumpyio as fnio

### RUN SPECIFIC CONFIGURATION ###
data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/'
test_loop = False        # test loop with single sample for debugging


if __name__ == '__main__':
    # creating a list of dictionaries of all samples from train, val and test
    val_list = create_bids_path_list_of_dicts(data_dir + '/val')
    test_list = create_bids_path_list_of_dicts(data_dir + '/test')
    train_list = create_bids_path_list_of_dicts(data_dir + '/train')
    list_of_dicts = val_list + test_list + train_list 

    sample_path = list_of_dicts[0]['seg']
    sample = NII.load(sample_path, seg=True)    # loading a sample

    max_dims = sample.shape # Determining the dimension of the images

    # Determining the center of the images
    center = np.zeros(3, dtype=int)
    for i, coordinate in  enumerate(max_dims):
        center[i] = coordinate // 2

    min_crop_dims = np.zeros(3, dtype=int)  #creating empty variable for the minimum viable crop

    for iteration, subject in enumerate(list_of_dicts):
        #loading segmentation for subject
        seg_path = subject['seg']
        seg = NII.load(seg_path, seg=True)

        #determining min crop of segmentation
        crop = NII.compute_crop(seg)

        # checking which slicing index is closer to the boundary of the image to have a symmetric crop
        crop_indices = np.zeros(3, dtype=int)
        for i, slice in enumerate(crop):
            dist_start = slice.start
            dist_stop = max_dims[i] - (slice.stop + 1)  # slice.stop still contains part of the segmentation -> +1
            if dist_start > dist_stop:
                crop_indices[i] = (slice.stop + 1)  # slice.stop still contains part of the segmentation -> +1
            else:
                crop_indices[i] = slice.start

        #calculating the necessary center crop i.e. size of image symmetric around the center
        center_crop = np.zeros(3, dtype=int)
        for  i, coordinate  in enumerate(crop_indices):
                center_crop[i] = max(2*(coordinate - center[i]),2*(center[i] - coordinate)) 

        # updating the min_crop_dims
        for idx, crop_dim in enumerate(center_crop):
            if crop_dim > min_crop_dims[idx]:
                min_crop_dims[idx] = crop_dim

        # save the center crop for each subject to a csv
        with open(data_dir + 'crop_dims.csv', 'a') as f:
            f.write(f"{subject['subject']},{center_crop[0]},{center_crop[1]},{center_crop[2]}\n")

        # # printing for debugging
        # print(f"Subject: {subject['subject']}")
        # print(f"seg.shape: {seg.shape}")
        # print(f"max_dims: {max_dims}")
        # print(f"center: {center}")
        # print(f"crop: {crop}")
        # print(f"crop_indices: {crop_indices}")
        # print(f"min_crop_dims: {min_crop_dims}")
        # print(f"center_crop: {center_crop}")
        # print(f"iteration: {iteration}")
        # # testing single sample

        # #plotting for debugging
        # #cropping the segmentation and slicing in the center
        # cropped_seg = seg.apply_center_crop(center_crop)
        # cropped_seg_array = cropped_seg.get_array()
        # cropped_seg_slice = get_central_slice(cropped_seg_array)
        # print(f"cropped_seg_slice.shape: {cropped_seg_slice.shape}")

        # #cropping the image and slicing in the center
        # img_path = subject['t1c']
        # img = NII.load(img_path, seg=False)
        # cropped_img = img.apply_center_crop(center_crop)
        # cropped_img_array = cropped_img.get_array()
        # cropped_img_slice = get_central_slice(cropped_img_array)
        # print(f"cropped_img_slice.shape: {cropped_img_slice.shape}")

        # #plotting the slices
        # plot_slices(cropped_img_slice, cropped_seg_slice)

        if test_loop:     
            # for testing purposes -> exit after one iteration
            break

    # save the min_crop_dims to a csv
    with open(data_dir + 'min_crop_dims.csv', 'w') as f:
        f.write(f"{min_crop_dims[0]},{min_crop_dims[1]},{min_crop_dims[2]}\n")

