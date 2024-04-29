import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

def create_bids_dict(root_dir):
    """
    Scan the specified directory for MRI and segmentation files organized in a BIDS-like structure.

    Args:
    root_dir (str): The root directory containing subdirectories for each subject.

    Returns:
    dict: A dictionary where each key is a subject directory name, and the value is another
          dictionary with keys for each MRI type and the segmentation mask, containing their file paths.
    """
    mri_types = ['t1c', 't1n', 't2f', 't2w', 'seg']
    data_dict = {}

    # Iterate through each directory in the root directory
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            #Initialize a sub-dictionary for each subject
            subject_files = {mri_type: None for mri_type in mri_types}

            # Search for MRI and segmentation files
            for file in os.listdir(folder_path):
                for mri_type in mri_types:
                    if f"{folder}-{mri_type}.nii.gz" in file:
                        subject_files[mri_type] = os.path.join(folder_path, file)

            # Add the dictionary for the current subject to the main dictionary
            data_dict[folder] = subject_files

    return data_dict

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
    #Is this what we want? Do we want the data normalized over the whole dataset or per image? Per Image, due to bias on contrast from MRI to MRI
    mean = np.mean(slice_array)
    std = np.std(slice_array)
    normalized_slice = (slice_array - mean) / std if std > 0 else slice_array
    return normalized_slice


class BraTSDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        """
        Parameters:
        image_paths(): dict of subject number and corresponding file paths to the relevant mri images
        mask_paths(): dict of subject number and correspondiong file path to relevant segmentation mask
        transform(): transformation of image data
        """

        self.image_paths = image_paths        
        self.mask_paths =mask_paths
        self.transform = transform

    def __len(self):
        return len()
    
    def __getitem__(self, idx):
        image = load_nifti_as_array(self.image_paths[idx])
        mask = load_nifti_as_array(self.mask_paths[idx])

        image_slice = get_central_slice(image)
        mask_slice = get_central_slice(mask)

        image_slice = normalize(image_slice)
        mask_slice = np.array(mask_slice, dtype=np.int64)   # Ensure mask is integery type

        if self.transform:
            image_slice = self.transform(image_slice)
            mask_slice = self.transform(mask_slice)

        # Add a channel dimension and convert to tensors
        image_slice = torch.from_numpy(image_slice).unsqueeze(0).float()
        mask_slice = torch.from_numpy(mask_slice).unsqueeze(0).long()
        
        return image_slice, mask_slice
    
