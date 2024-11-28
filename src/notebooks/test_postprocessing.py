# %%
# import sys
# import os
# from pathlib import Path

# file = Path(__file__).resolve()

#sys.path.append(str(file.parents[0]))
#sys.path.append(str(file.parents[1]))
# sys.path.append(str(file.parents[2]))

# %%
from pathlib import Path
import torch
import numpy as np
#from utils.brats_tools import postprocessing, get_central_slice, plot_slices
from TPTBox import NII
import matplotlib.pyplot as plt
# %%
def get_central_slice(nifti_np_array: np.ndarray, axis:int=2)->np.ndarray:
    """
    Export the central slice of a NIfTI image along the specified axis.
    
    Parameters:
        nifti_np_array(): NIfTI image that has already been exported to a NumPy array. Expects array of dim [x, y, z].
        axis (int): Axis along which the central slice is to be taken (0, 1, or 2).
                   0 - Sagittal, 1 - Coronal, 2 - Axial
    Returns:
        numpy.ndarray: The central slice as a NumPy array.
    """
    if nifti_np_array.ndim == 3:
        # Determine the slice index
        slice_index = nifti_np_array.shape[axis] // 2
        
        # Extract the central slice based on the axis
        if axis == 0:
            slice_data = nifti_np_array[slice_index, :, :]
        elif axis == 1:
            slice_data = nifti_np_array[:, slice_index, :]
        elif axis == 2:
            slice_data = nifti_np_array[:, :, slice_index]
        else:
            raise ValueError("Invalid axis. Axis must be 0, 1, or 2.")
    elif nifti_np_array.ndim == 4:
        # Determine the slice index
        slice_index = nifti_np_array.shape[axis+1] // 2

        # Extract the central slice based on the axis
        if axis == 0:
            slice_data = nifti_np_array[: ,slice_index, :, :]
        elif axis == 1:
            slice_data = nifti_np_array[:,:, slice_index, :]
        elif axis == 2:
            slice_data = nifti_np_array[:, :, :, slice_index]
        else:
            raise ValueError("Invalid axis. Axis must be 0, 1, or 2.")

    return slice_data  
# %%
def plot_slices(mri_slice, seg_slice, plt_title:str="" , omit_background=True, show=True, save_path=None, cmap_mri='gray', cmap_seg='jet'):

    masked_seg_slice = np.where(seg_slice == 0, np.nan, seg_slice)  # Replace 0s with NaN, leave everything else unchanged

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(mri_slice.T, cmap=cmap_mri, origin='lower')
    ax[1].imshow(mri_slice.T, cmap=cmap_mri, origin='lower')

    # Give Titles to the plots
    ax[0].set_title('MRI Slice')
    ax[1].set_title('Segmentation Slice')

    if plt_title:
        plt.suptitle(plt_title, fontsize=30)

    if omit_background==True:
        # Only overlay areas where mask is True, with the segmentation mask using a colormap
        ax[1].imshow(masked_seg_slice.T, cmap=cmap_seg, alpha=0.5, origin='lower')
    else:
        ax[1].imshow(seg_slice.T, cmap=cmap_seg, alpha = 0.5, origin='lower')      #Overlay with transparency

    if save_path:
        plt.savefig(save_path)  # Save the figure to the specified path

    if show:
        plt.show()      # Show the plot if requested
    else:
        plt.close(fig)  # Only close the figure if not showing it
# %%
def postprocessing(soft_scores, dec: int = 3, smooth_to_zero: bool = False):
    """
    Applies post-processing to scores:
    1. Ensures all values are >= 0 (using ReLU for PyTorch and np.maximum for NumPy).
    2. Rounds to a specified number of decimals.
    3. Optionally smooths values below a threshold (to 0) and above a threshold (to 1).

    Parameters:
        soft_scores (np.ndarray or torch.Tensor): Input scores.
        dec (int): Number of decimals to round to.
        smooth_to_zero (bool): Whether to apply smoothing for values < 0.05 or > 0.95.

    Returns:
        np.ndarray or torch.Tensor: Processed scores.
    """
    # Ensure all values are >= 0
    if isinstance(soft_scores, np.ndarray):
        soft_scores = np.maximum(0, soft_scores)
    elif isinstance(soft_scores, torch.Tensor):
        soft_scores = torch.nn.functional.relu(soft_scores)
    
    soft_scores = soft_scores.round(decimals = dec)
    
    # Optional smoothing (applies to both NumPy and PyTorch)
    if smooth_to_zero:
        soft_scores[soft_scores < 0.05] = 0.0
        soft_scores[soft_scores > 0.95] = 1.0

    return soft_scores
# %%
b, c, h, w, d = 1, 2, 2, 2, 2
#tensor = torch.rand((b,c,h,w,d))
tensor = torch.FloatTensor((b,c,h,w,d)).uniform_(-1, 1)
np_array = tensor.numpy()
print(tensor)
# %%
print(postprocessing(tensor, 3, True))
print(postprocessing(np_array, 3, True))
# %%
## Test on actual prediction
pred_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/preds/3D_UNet_v11__exp1_mse_high_sigma_linear_test_set/BraTS-GLI-00108-000/channelwise/niftis/BraTS-GLI-00108-000-prob-class3-_exp1_mse_high_sigma_linear_test_set.nii.gz"
pred_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/preds/3D_UNet_v11__exp1_mse_high_sigma_linear_test_set/BraTS-GLI-00014-000/channelwise/niftis/BraTS-GLI-00014-000-prob-class3-_exp1_mse_high_sigma_linear_test_set.nii.gz"
img_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test/BraTS-GLI-00014-000/BraTS-GLI-00014-000-t2f.nii.gz"
pred = NII.load(pred_path, seg=False)
img = NII.load(img_path, seg = False)
# %%
pred_array = pred.get_array()
img_array = img.get_array()
print(np.unique(pred_array, return_counts=True))
# %%
img_slice = get_central_slice(img_array)
pred_slice = get_central_slice(pred_array)
plot_slices(img_slice,postprocessing(pred_slice, 3, True))
# %%
print(np.unique(pred_slice, return_counts=True))
print(np.unique(postprocessing(pred_slice, 3, False), return_counts=True))
# %%
