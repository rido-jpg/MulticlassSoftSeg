import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def plot_slices(mri_slice, seg_slice, plt_title:str="" , omit_background=True, show=True, save_path=None):
    # Create a masked array where only values 1, 2, and 3 are included
    # and other values are set to be transparent
    mask = np.isin(seg_slice, [1, 2, 3])
    masked_seg_slice = np.where(mask, seg_slice, np.nan)  # Replace 0s with NaN for transparency

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ax[0].imshow(mri_slice.T, cmap='gray', origin='lower')
    ax[1].imshow(mri_slice.T, cmap='gray', origin='lower')

    # Give Titles to the plots
    ax[0].set_title('MRI Slice')
    ax[1].set_title('Segmentation Slice')

    if plt_title:
        plt.suptitle(plt_title, fontsize=30)

    if omit_background==True:
        # Only overlay areas where mask is True, with the segmentation mask using a colormap
        ax[1].imshow(masked_seg_slice.T, cmap='jet', alpha=0.5, origin='lower')
    else:
        ax[1].imshow(seg_slice.T, cmap='jet', alpha = 0.5, origin='lower')      #Overlay with transparency

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

    return slice_data    

def preprocess(img:np.array, seg:bool, binary:bool) -> torch.Tensor:
    #Normalize the MRI image
    if not seg:
        img = normalize(img)

    # Convert segmentation mask to binary classification labels (tumor vs. non-tumor)
    if binary and seg:
        img[img > 0] = 1

    # Convert numpy arrays to PyTorch tensors
    img = torch.from_numpy(img)

    # # Add a channel dimension if it's missing
    # if img.ndim == 2:  # if the image is 2D
    #     img = img.unsqueeze(0)
    # elif img.ndim == 3:  # if the image is 3D
    #     img = img.unsqueeze(0)

    img = img.unsqueeze(0)  # add a channel dimension

    # Ensure the image is in the correct type
    if not seg:
        img = img.float()
    else:
        img = img.long()

    return img

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
    if std > 0:
        normalized_array = nifti_array.copy()  # Create a copy to retain the original zero values
        normalized_array[mask] = (masked_array - mean) / std
    else:
        raise ValueError("Standard deviation of the masked elements is zero, normalization cannot be performed.")
    
    return normalized_array