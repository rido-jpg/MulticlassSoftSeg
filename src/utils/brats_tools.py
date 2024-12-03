import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import scipy.ndimage
from argparse import Namespace
from TPTBox import NII, np_utils
import utils.fastnumpyio.fastnumpyio as fnio

def plot_slices(mri_slice, seg_slice, plt_title:str="" , omit_background=True, show=True, save_path=None, cmap_mri='gray', cmap_seg='jet', gt_slice = None):

    masked_seg_slice = np.where(seg_slice == 0, np.nan, seg_slice)  # Replace 0s with NaN, leave everything else unchanged

    if gt_slice is not None:
        masked_gt_slice = np.where(gt_slice == 0, np.nan, gt_slice)  # Replace 0s with NaN, leave everything else unchanged

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(mri_slice.T, cmap=cmap_mri, origin='lower')
    ax[1].imshow(mri_slice.T, cmap=cmap_mri, origin='lower')

    # Give Titles to the plots
    if gt_slice == None:
        ax[0].set_title('MRI Slice')
        ax[1].set_title('Segmentation Slice')
    else:
        ax[0].set_title('Ground Truth Slice')
        ax[1].set_title('Prediction Slice')

    if plt_title:
        plt.suptitle(plt_title, fontsize=30)

    if omit_background==True:
        # Only overlay areas where mask is True, with the segmentation mask using a colormap
        ax[1].imshow(masked_seg_slice.T, cmap=cmap_seg, alpha=0.5, origin='lower')
        if gt_slice is not None:
            ax[0].imshow(masked_gt_slice.T, cmap=cmap_seg, alpha=0.5, origin='lower')
    else:
        ax[1].imshow(seg_slice.T, cmap=cmap_seg, alpha = 0.5, origin='lower')      #Overlay with transparency
        if gt_slice is not None:
            ax[0].imshow(gt_slice.T, cmap=cmap_seg, alpha=0.5, origin='lower')

    if save_path:
        plt.savefig(save_path)  # Save the figure to the specified path

    if show:
        plt.show()      # Show the plot if requested
    else:
        plt.close(fig)  # Only close the figure if not showing it

def get_central_slice_from_tensor(nifti_tensor: torch.Tensor, axis:int=2)->torch.Tensor:
    """
    Export the central slice of a NIfTI image tensor along the specified axis.
    
    Parameters:
        nifti_tensor(): NIfTI image that has already been exported to a NumPy array. Expects array of dim [C ,x, y, z].
        axis (int): Axis along which the central slice is to be taken (0, 1, or 2).
                   0 - Sagittal, 1 - Coronal, 2 - Axial
    Returns:
        torch.Tensor: The central slice as a NumPy array.
    """
    # Determine the slice index
    slice_index = nifti_tensor.shape[axis+1] // 2
    
    # Extract the central slice based on the axis
    if axis == 0:
        slice_data = nifti_tensor[: ,slice_index, :, :]
    elif axis == 1:
        slice_data = nifti_tensor[:,:, slice_index, :]
    elif axis == 2:
        slice_data = nifti_tensor[:, :, :, slice_index]
    else:
        raise ValueError("Invalid axis. Axis must be 0, 1, or 2.")

    return slice_data

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

def preprocess(img_np:np.array, seg:bool, binary:bool, n_classes:int=None, opt: Namespace=None, soft:bool=False) -> torch.Tensor:
    '''
    Preprocess the MRI image and segmentation mask.
    1. Z-Standardize the MRI intensity values
    2. Depending on config convert the segmentation mask to binary classification labels (tumor vs. non-tumor)
    3. Convert numpy arrays to PyTorch tensors (float for MRI, long for segmentation mask)
    4. If soft, one-hot encode, premute and soften the segmentation mask. The resulting shape is [C, H, W, D], where C=n_classes
        Else, just unsqueeze and add Channel dimension to achieve desired shape [C, H, W, D] where C=1

    Parameters:
        img (np.array): MRI image or segmentation mask
        seg (bool): True if the input is a segmentation mask
        binary (bool): True if the segmentation mask should be converted to binary labels
        n_classes (int): Number of classes in the segmentation mask
        opt (Namespace): Configuration containing, soft parameter, sigma, and dilate parameter
    
    '''
    #Normalize the MRI image
    if not seg:
        img_np = normalize(img_np)

    # Convert segmentation mask to binary classification labels (tumor vs. non-tumor)
    if binary and seg:
        img_np = (img_np > 0)

    # Convert numpy arrays to PyTorch tensors
    img_tensor = torch.from_numpy(img_np.copy())
    # img_tensor = torch.tensor(img) 
    """
    UserWarning: The given NumPy array is not writable, and PyTorch does
    not support non-writable tensors. This means writing to this tensor will result in undefined behavior. You may want to copy the array to protect its data or make
    it writable before converting it to a tensor. This type of warning will be suppressed for the rest of this program. (Triggered internally at /opt/conda/conda-bld
    /pytorch_1712608853085/work/torch/csrc/utils/tensor_numpy.cpp:206.)
    """

    # Ensure the image has the correct type
    if not seg:
        img_tensor = img_tensor.float()
    else:
        img_tensor = img_tensor.long()

    # In Soft Case, one-hot encode the segmentation mask and apply Gaussian smoothing, then permute to [C, H, W, D], else just add channel dimension
    if soft:
        img_tensor = F.one_hot(img_tensor, num_classes=n_classes).permute(3, 0, 1, 2) #one-hot encode and permute to [C, H, W, D]

        if opt.dilate != 0:
            img_tensor = dilate_gt(img_tensor, opt.dilate)

        img_tensor = soften_gt(img_tensor, opt.sigma)   # Apply Gaussian smoothing to the segmentation mask

        if get_option(opt, "soft_gt_norm", None) is not None:
            if opt.soft_gt_norm == "l1_norm":
                img_tensor = l1_norm(img_tensor, dim = 0)
            elif opt.soft_gt_norm == "temp_scaled_softmax":
                img_tensor = temperature_scaled_softmax(img_tensor, dim = 0, temperature=opt.softmax_temperature)
        
        if get_option(opt, "round", None) is not None:
            img_tensor = img_tensor.round(decimals = opt.round)

    else:
        img_tensor = img_tensor.unsqueeze(0)    # Add channel dimension to achieve desired shape [C, H, W, D]

    return img_tensor

def slice_and_pad(img:torch.Tensor, padding:tuple[int, int]|tuple[int, int, int]) -> torch.Tensor:
    # Get a 2D slice
    img= get_central_slice_from_tensor(img)

    if padding:    # Move to transform?
        # Calculate the padding required to achieve the desired size
        x, y = img.shape[-2:]
        target_x, target_y = padding
        pad_x = max(0, target_x - x)
        pad_y = max(0, target_y - y)
        
        # Apply padding to the image
        img = F.pad(img, (0, pad_x, 0, pad_y))
    return img

def brats_load(path:str, format:str, seg:bool=False) -> np.ndarray:
    '''
    Load a NIfTI or fnio file from the specified path.
    
    Parameters:
        path (str): Path to the img file
        format (str): Format of the NIfTI file (e.g. 'nii' or 'nii.gz')
        seg (bool): True if the file is a segmentation mask
    Returns:
        np.ndarray: NIfTI image as a NumPy array
    '''
    if format == 'fnio':
        img_np_array = fnio.load(str(path))

    elif format == 'nii.gz':
        img_np_array = load_nifti_as_array(str(path), seg)

    return img_np_array

def normalize(nifti_array:np.ndarray) -> np.ndarray:
    # Z-Score standardization
    # Mask to exclude zeros
    mask = nifti_array != 0
    masked_array = nifti_array[mask]
    
    if masked_array.size == 0:
        raise ValueError("No non-zero elements found in the array for normalization.")

    mean = masked_array.mean()
    std = masked_array.std()
    
    # Only apply normalization to non-zero elements
    normalized_array = nifti_array.copy()  # Create a copy to retain the original zero values
    normalized_array[mask] = (masked_array - mean) / max(std, 1e-8)
    
    return normalized_array

def soften_gt(gt:torch.Tensor, sigma:float=1.0) -> torch.Tensor:
    '''
    Soften the ground truth segmentation mask using a Gaussian kernel.
    Parameters:
        gt (torch.Tensor): Ground truth segmentation mask in one-hot encoded format [C, H, W, D]
        sigma (float): Standard deviation of the Gaussian kernel
    Returns:
        torch.Tensor: Softened segmentation mask in [C, H, W, D] format
    '''
    # Apply Gaussian filter to each channel
    filtered_gt = torch.empty_like(gt, dtype=torch.float32)  # Create an empty tensor to store the results

    for i in range(filtered_gt.shape[0]):  # Loop through each class channel
        filtered_gt[i] = torch.tensor(scipy.ndimage.gaussian_filter(gt[i].numpy().astype(float), sigma=sigma))
    
    return filtered_gt

def dilate_gt(gt_tensor:torch.Tensor, layers:int) -> torch.Tensor:
    '''
    Dilate the ground truth segmentation mask using np_utils_dilate_msk from TPTBox.
    Parameters:
        gt (torch.Tensor): Ground truth segmentation mask in one-hot encoded format [C, H, W, D]
        mm (int): Number of voxel neighbor layers to dilate to (assuming resolution of 1mm)
    Returns:
        torch.Tensor: Dilated segmentation mask in [C, H, W, D] format
    '''
    oh_gt_np = gt_tensor.numpy()  # Convert the PyTorch tensor to a NumPy array
    dilated_oh_gt_np = np.zeros_like(oh_gt_np)    # Create an empty array to store the dilated mask

    for channel in range(oh_gt_np.shape[0]):
        dilated_oh_gt_np[channel] = np_utils.np_dilate_msk(oh_gt_np[channel], mm=layers, connectivity=3)

    dilated_gt_tensor = torch.from_numpy(dilated_oh_gt_np)
    
    del oh_gt_np, dilated_oh_gt_np  # Delete the NumPy arrays to free up memory
    
    return dilated_gt_tensor

def load_nifti_as_array(file_path: str, seg:bool=False)->np.ndarray:
    #Load nifti file and convert to np array
    nifti_image = NII.load(file_path, seg)
    nifti_np_array = nifti_image.get_array()
    return np.ascontiguousarray(nifti_np_array) # Ensure C-contiguity for fast numpy io

def temperature_scaled_softmax(tensor, dim:int = 0, temperature=1.0):
    """
    Higher Temperature -> less squashing
    """
    tensor = tensor / temperature
    return torch.softmax(tensor, dim=dim)

def relu_l1_norm(logits, dim = 0):
    relu_output = F.relu(logits)  # Apply ReLU
    l1_norm = torch.linalg.vector_norm(relu_output, ord=1, dim=dim, keepdim=True)  # Compute L1 norm along C
    l1_norm = torch.where(l1_norm == 0, torch.ones_like(l1_norm), l1_norm)  # Avoid division by zero
    return relu_output / l1_norm

def l1_norm(probs, dim:int = 0):
    return probs / probs.sum(dim=dim, keepdim=True)

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

def postprocessing(soft_scores, dec: int = 3, smooth_to_zero: bool = False):
    """
    Applies post-processing to scores:
    1. Ensures all values are >= 0 (using ReLU for PyTorch and np.maximum for NumPy).
    2. Rounds to a specified number of decimals.
    3. Optionally smooths values below a threshold of 0.05 (to 0) and above a threshold of 0.95 (to 1).

    Parameters:
        soft_scores (np.ndarray or torch.Tensor): Input scores.
        dec (int): Number of decimals to round to.
        smooth_to_zero (bool): Whether to apply smoothing for values < 0.05 and > 0.95.

    Returns:
        np.ndarray or torch.Tensor: Processed scores.
    """
    # Ensure all values are >= 0
    if isinstance(soft_scores, np.ndarray):
        soft_scores = np.maximum(0, soft_scores)
    elif isinstance(soft_scores, torch.Tensor):
        soft_scores = F.relu(soft_scores)
    
    soft_scores = soft_scores.round(decimals = dec)
    
    # Optional smoothing (applies to both NumPy and PyTorch)
    if smooth_to_zero:
        soft_scores[soft_scores < 0.05] = 0.0
        soft_scores[soft_scores > 0.95] = 1.0

    return soft_scores