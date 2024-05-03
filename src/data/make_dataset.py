import os
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

def create_bids_path_list_of_dicts(root_dir, prefix=""):
    """
    Scan the specified directory for MRI and segmentation files organized in a BIDS-like structure.

    Args:
    root_dir (str): The root directory containing subdirectories for each subject.

    prefix (str): if all folders have the same prefix e.g. "BraTS-GLI-" you can add it to be excluded from the dict key

    Returns:
    dict: A dictionary where each key is a subject directory name, and the value is another
          dictionary with keys for each MRI type and the segmentation mask, containing their file paths.
    """
    mri_types = ['t1c', 't1n', 't2f', 't2w', 'seg']
    list_of_path_dicts = []
    root_path = Path(root_dir)

    # Iterate through each directory in the root directory
    for folder_path in root_path.iterdir():
        if folder_path.is_dir():
            #Initialize a sub-dictionary for each subject
            folder = folder_path.name
            modified_folder_name = folder.replace(prefix, "") if folder.startswith(prefix) else folder
            subject_files = {mri_type: None for mri_type in mri_types}

            # Search for MRI and segmentation files
            for file_path in folder_path.iterdir():
                file_name = file_path.name
                for mri_type in mri_types:
                    if f"{folder}-{mri_type}.nii.gz" in file_name:
                        subject_files[mri_type] = file_path.absolute()

            # Add the dictionary for the current subject to the main list
            subject_dict = {'subject': modified_folder_name, **subject_files}
            list_of_path_dicts.append(subject_dict)

    return list_of_path_dicts

def create_bids_array_list_of_dicts(root_dir, prefix=""):
    list_of_path_dicts = create_bids_path_list_of_dicts(root_dir, prefix)
    list_of_array_dicts = []

    for subject_dict in list_of_path_dicts:
        subject = subject_dict['subject']
        array_dict = {'subject': subject}
        for contrast in ['t1c', 't1n', 't2f', 't2w', 'seg']: # List MRI types
            filepath = subject_dict.get(contrast)
            if filepath and Path(filepath).exists():  # Ensure the file exists
                if contrast == 'seg':
                    array_dict[contrast] = get_central_slice(load_nifti_as_array(filepath))
                else: 
                    array_dict[contrast] = load_normalized_central_slice_as_array(filepath)
            else:
                array_dict[contrast] = None  # Handle missing or inaccessible files gracefully

        # Add the dictionary for the current subject to the main list
        list_of_array_dicts.append(array_dict)

    return list_of_array_dicts

def load_nifti_as_array(file_path):
    #Load nifti file and convert to np array
    nifti_image = nib.load(file_path)
    nifti_np_array = np.array(nifti_image.get_fdata(), dtype=np.float64)
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

def normalize(nifti_array):
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

def load_normalized_central_slice_as_array(file_path):
    nifti_array = load_nifti_as_array(file_path)
    normalized_array = normalize(nifti_array)
    central_slice = get_central_slice(normalized_array)
    return central_slice

class BidsDataset(Dataset):
    def __init__(self, root_dir, prefix="", contrast='t2f', binary=True, transform=None):
        """
        Parameters:
        root_dir (str): The root directory containing subdirectories for each subject.
        prefix (str): if all folders have the same prefix e.g. "BraTS-GLI-" you can add it to be excluded from the dict key
        contrast (str): type of mri contrast you want to use for the dataset, options: 't1c', 't1n', 't2f', 't2w'
        binary (bool): if True, convert segmentation mask to binary classification labels (tumor vs. non-tumor)
        transform(): transformation of image data
        """

        self.root_dir = root_dir   
        self.prefix = prefix     
        self.contrast = contrast
        self.transform = transform
        self.binary = binary

        self.bids_list = create_bids_array_list_of_dicts(self.root_dir, prefix=self.prefix)
        
    def __len__(self):
        return len(self.bids_list)
    
    def __getitem__(self, idx):
        img = self.bids_list[idx][self.contrast]
        seg = self.bids_list[idx]['seg']

        seg = np.array(seg, dtype=np.int64)   # Ensure mask is integer type

        # Convert segmentation mask to binary classification labels (tumor vs. non-tumor)
        if self.binary:
            seg[seg > 0] = 1

        if self.transform:
            img, seg = self.transform(img, seg)

        # Add a channel dimension and convert to tensors
        img = torch.from_numpy(img).unsqueeze(0).float()
        seg = torch.from_numpy(seg).unsqueeze(0).long()
        
        return img, seg
    
