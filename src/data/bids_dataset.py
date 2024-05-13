import os
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import random
import csv
import sys

from TPTBox import NII
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

# os.chdir(os.path.expanduser('~/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg'))
from external.fastnumpyio.fastnumpyio import save, load, pack, unpack

# Array of all MRI contrasts and the segmentation mask  
contrasts = ['t1c', 't1n', 't2f', 't2w', 'seg']

class BidsDataset(Dataset):
    def __init__(self, root_dir:str, prefix:str="", contrast:str='t2f', suffix:str='fnio', do2D:bool=True, binary:bool=True, transform=None, padding=(256, 256)):
        """
        Parameters:
        root_dir (str): The root directory containing subdirectories for each subject.
        prefix (str): if all folders have the same prefix e.g. "BraTS-GLI-" you can add it to be excluded from the dict key
        contrast (str): type of mri contrast you want to use for the dataset, options: 't1c', 't1n', 't2f', 't2w'
        suffix (str): The file extension of the MRI and segmentation files. Can be modified if we saved the files in different format than .nii.gz, e.g. in fast numpy io format "fnio"
        do2D (bool): if True, use 2D slices of the MRI images and segmentation masks
        binary (bool): if True, convert segmentation mask to binary classification labels (tumor vs. non-tumor)
        transform(): transformation of image data
        """

        self.root_dir = root_dir   
        self.prefix = prefix     
        self.contrast = contrast
        self.suffix = suffix
        self.do2D = do2D
        self.transform = transform
        self.binary = binary
        self.padding = padding

        self.bids_list = create_bids_path_list_of_dicts(self.root_dir, prefix=self.prefix, suffix=self.suffix)
        
    def __len__(self):
        return len(self.bids_list)
    
    def __getitem__(self, idx):
        img_path = self.bids_list[idx][self.contrast]
        mask_path = self.bids_list[idx]['seg']

        # Load the MRI image and segmentation mask
        if self.suffix == 'fnio':
            img = load(str(img_path))
            mask = load(str(mask_path))

        elif self.suffix == 'nii.gz':
            img = load_nifti_as_array(str(img_path))
            mask = load_nifti_as_array(str(mask_path))

        mask = np.array(mask, dtype=np.int64)   # Ensure mask is integer type

        # Normalize the MRI image
        img = normalize(img)  
        
        # Get a 2D slice if specified
        if self.do2D:
            #img = img.squeeze(0)
            #mask = mask.squeeze(0)
            img, mask = get_central_slice(img), get_central_slice (mask)      

        # Convert segmentation mask to binary classification labels (tumor vs. non-tumor)
        if self.binary:
            mask[mask > 0] = 1

        # Convert numpy arrays to PyTorch tensors and add a channel dimension
        img = torch.from_numpy(img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).long()

        if self.padding:
            # Calculate the padding required to achieve the desired size
            h, w = img.shape[-2:]
            target_h, target_w = self.padding
            pad_h = max(0, target_h - h)
            pad_w = max(0, target_w - w)
            
            # Apply padding to the image and segmentation mask
            img = F.pad(img, (0, pad_w, 0, pad_h))
            mask = F.pad(mask, (0, pad_w, 0, pad_h))
        
        if self.transform:
            img, mask = self.transform(img, mask)
        
        return img, mask

def create_bids_path_list_of_dicts(root_dir:str, prefix:str="", suffix:str="nii.gz")->list:
    """
    Scan the specified directory for MRI and segmentation files organized in a BIDS-like structure.

    Args:
    root_dir (str): The root directory containing subdirectories for each subject.

    prefix (str): if all folders have the same prefix e.g. "BraTS-GLI-" you can add it to be excluded from the dict key

    suffix (str): The file extension of the MRI and segmentation files. Can be modified if we saved the files in different format than .nii.gz, e.g. in fast numpy io format "fnio"

    Returns:
    dict: A dictionary where each key is a subject directory name, and the value is another
          dictionary with keys for each MRI type and the segmentation mask, containing their file paths.
    """
    list_of_path_dicts = []
    root_path = Path(root_dir)

    # Iterate through each directory in the root directory
    for folder_path in root_path.iterdir():
        if folder_path.is_dir():
            # Initialize a sub-dictionary for each subject
            folder = folder_path.name
            modified_folder_name = folder.replace(prefix, "") if folder.startswith(prefix) else folder
            subject_files = {contrast: None for contrast in contrasts}

            # Search for MRI and segmentation files
            for file_path in folder_path.iterdir():
                file_name = file_path.name
                for contrast in contrasts:
                    if f"{folder}-{contrast}.{suffix}" in file_name:
                        subject_files[contrast] = file_path.absolute()

            # Add the dictionary for the current subject to the main list
            subject_dict = {'subject': modified_folder_name, **subject_files}
            list_of_path_dicts.append(subject_dict)

    return list_of_path_dicts

def create_bids_array_list_of_dicts(root_dir:str, prefix:str= "") -> list: #  not needed anymore ?
    list_of_path_dicts = create_bids_path_list_of_dicts(root_dir, prefix)
    list_of_array_dicts = []

    for subject_dict in list_of_path_dicts:
        subject = subject_dict['subject']
        array_dict = {'subject': subject}
        for contrast in contrasts:  # MRI types from the contrasts list
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

def load_nifti_as_array(file_path: str):
    #Load nifti file and convert to np array
    nifti_image = NII.load(file_path)
    nifti_np_array = nifti_image.get_array()
    return np.ascontiguousarray(nifti_np_array) # Ensure C-contiguity for fast numpy io

def get_central_slice(nifti_np_array: np.ndarray, axis:int=2)->np.ndarray:
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

def load_normalized_central_slice_as_array(file_path: str)->np.ndarray:
    nifti_array = load_nifti_as_array(file_path)
    normalized_array = normalize(nifti_array)
    central_slice = get_central_slice(normalized_array)
    return central_slice   

def split_data_and_save_to_csv(data: list, split_percentage: float, directory: str, train_val:bool = True, seed: int = 42):
    """
    Split the data into sets for either training and testing or training and validation and save the "subject" values to CSV files.
    Parameters:
    data (list): The list of data dictionaries to be split.
    split_percentage (float): The percentage of data to be used for training.
    directory (str): The directory in which to save the CSV files.
    train_val (bool): If True, split the data into training and validation sets. If False, split the data into training and testing sets.
    seed (int): The seed for the random number generator.
    """
    # Set the seed for reproducibility
    random.seed(seed)

    # Shuffle the data
    random.shuffle(data)

    # Calculate the index at which to split the data
    split_index = int(len(data) * split_percentage)

    # Check if the data should be split into training and validation or training and testing sets
    if train_val:
        _split_and_save(True, data, split_index, directory)

    else:
        _split_and_save(False, data, split_index, directory)

def _split_and_save(val_split: bool, data: list, split_index: int, directory: str):
        """"
        Parameters:
        val_split (bool): If True, split the data into training and validation sets. If False, split the data into training and testing sets.
        """
        if val_split:
            rest = 'val'
        else:
            rest = 'test'

        # Split the data into training and validation/test sets
        train_data = [{'split': 'train', 'subject': d.get('subject', '')} for d in data[:split_index]]
        rest_data = [{'split': rest, 'subject': d.get('subject', '')} for d in data[split_index:]]

        # Create the directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)

        # Save the "subject" values from the training and validation/test sets to CSV files
        _save_to_csv(train_data + rest_data, Path(directory) / f"train_{rest}_split.csv")

def _save_to_csv(data: list, filename: Path):
    with filename.open('w', newline='') as csvfile:
        fieldnames = ['split', 'subject']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for d in data:
            writer.writerow(d)

def convert_nifti_to_fnio(file: str):
    # Save the NIfTI file as a fnio array in the same directory, with the same name
    nifti_np_array = load_nifti_as_array(file)
    new_file_name = file.replace('.nii.gz', '.fnio')
    save(new_file_name, nifti_np_array)

def save_bids_niftis_as_fnio(list_of_path_dicts:list):
    for subject_dict in list_of_path_dicts:
        for contrast in contrasts:
            filepath = subject_dict.get(contrast)
            print(f"subject: {subject_dict.get('subject')} and contrast: {contrast}")
            #print(filepath)
            if filepath and Path(filepath).exists():  # Ensure the file exists
                path_as_string = str(filepath)
                convert_nifti_to_fnio(path_as_string)
        print(f"subject {subject_dict.get('subject')} done.")
        print(f"-------------------------------")