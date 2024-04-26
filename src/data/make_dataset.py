import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset

def load_nifti_as_array(file_path):
    #Load nifti file and convert to np array
    nifti_image = nib.load(file_path)
    nifti_np_array = np.array(nifti_image.get_fdata(), dtype=np.float32)
    return nifti_np_array

def get_central_slice(nifti_np_array, axis=2):
    """
    Export the central slice of a NIfTI image along the specified axis.
    
    Parameters:
        nifti_np_array(): NIfTI image that has already been exported to a NumPy array
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

def normalize(slice_array):
    #Is this what we want? We want the data to all be normalized between one and zero across the whole dataset, right?
    mean = np.mean(slice_array)
    std = np.std(slice_array)
    normalized_slice = (slice_array - mean) / std if std > 0 else slice_array
    return normalized_slice