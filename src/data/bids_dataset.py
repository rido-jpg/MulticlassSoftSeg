import os
import logging
#import nibabel as nib
import random
import csv
import sys
import math
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl


from TPTBox import NII
from torch import nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path

from monai.transforms import (
    SpatialPadd,
    CastToTyped,
    ResizeWithPadOrCropd,
    Compose,
)

file = Path(__file__).resolve()
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

import utils.fastnumpyio.fastnumpyio as fnio
from utils.brats_tools import preprocess, slice_and_pad, normalize, get_central_slice

# Configure logging
logging.basicConfig(level=logging.INFO)

# Array of all MRI contrasts and the segmentation mask  
contrasts = ['t1c', 't1n', 't2f', 't2w', 'seg']
modalities = ['t1c', 't1n', 't2f', 't2w']

# Keys of Dictionary that will be returned by the Dataset
brats_keys = ['img', 'seg']

# # Train Transforms
# train_transforms = Compose(
#     [
#         RandRotateD(keys=brats_keys, range_x=[- math.radians(30), math.radians(30)], prob=1, mode =["bilinear", "nearest"]),
#     ]
# )

class BidsDataModule(pl.LightningDataModule):
    def __init__(self, data_dir:str, contrast:str='t2f', format:str='fnio', do2D:bool=False, binary:bool=False, train_transform=None, test_transform=None, resize:tuple[int, int, int]=(200, 200, 152), batch_size:int=2, n_workers:int=16):
        super().__init__()
        self.data_dir = data_dir
        self.contrast = contrast
        self.format = format
        self.do2D = do2D
        self.binary = binary
        self.resize = resize
        self.batch_size = batch_size 
        self.train_transform = train_transform   
        self.test_transform = test_transform
        self.n_workers = n_workers

    def setup(self, stage: str = None) -> None:
        self.train_dataset = BidsDataset(self.data_dir +'/train', contrast=self.contrast, suffix=self.format, do2D=self.do2D, binary=self.binary, transform=self.train_transform,resize=self.resize)
        self.val_dataset = BidsDataset(self.data_dir + '/val', contrast=self.contrast, suffix=self.format, do2D=self.do2D, binary=self.binary, transform=self.test_transform,resize=self.resize)
        self.test_dataset = BidsDataset(self.data_dir + '/test', contrast=self.contrast, suffix=self.format, do2D=self.do2D, binary=self.binary, transform=self.test_transform,resize=self.resize)
        
    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.n_workers, pin_memory=True)
    
    def val_dataloader(self) -> torch.Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    def test_dataloader(self) -> torch.Any:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True)

class BidsDataset(Dataset):
    def __init__(self, data_dir:str, prefix:str="", contrast:str='t2f', suffix:str='fnio', do2D:bool=False, binary:bool=False, transform=None, resize:tuple[int, int, int]=(200, 200, 152)):
        """
        Parameters:
        data_dir (str): The data directory containing subdirectories for each subject.
        prefix (str): if all folders have the same prefix e.g. "BraTS-GLI-" you can add it to be excluded from the dict key
        contrast (str): type of mri contrast you want to use for the dataset, options: 't1c', 't1n', 't2f', 't2w', 'multimodal'
        suffix (str): The file extension of the MRI and segmentation files. Can be modified if we saved the files in different format than .nii.gz, e.g. in fast numpy io format "fnio" 
        do2D (bool): if True, use 2D slices of the MRI images and segmentation masks
        binary (bool): if True, convert segmentation mask to binary classification labels (tumor vs. non-tumor)
        transform(): transformation of image data (augmentations)
        resize (tuple): size of the image to be resized to
        """

        self.data_dir = data_dir   
        self.prefix = prefix     
        self.contrast = contrast
        self.suffix = suffix
        self.do2D = do2D
        self.transform = transform
        self.binary = binary
        #self.padding = padding
        self.resize = resize
        self.dict_keys = ['img', 'seg']
        self.img_key = self.dict_keys[0]
        self.seg_key = self.dict_keys[1]

        if do2D:
            self.resize = self.resize[:2]

        self.postprocess = Compose(
            [
                ResizeWithPadOrCropd(keys=brats_keys, spatial_size=self.resize, mode="symmetric"),
                CastToTyped(keys=brats_keys, dtype=(torch.float, torch.long)),
            ]
        )

        self.bids_list = create_bids_path_list_of_dicts(self.data_dir, prefix=self.prefix, suffix=self.suffix)
        
    def __len__(self):
        return len(self.bids_list)
    
    def __getitem__(self, idx):
        """"
        returns: dict with keys 'img' and 'seg' containing the MRI image and segmentation mask
        """
        # Loading segmentation mask
        mask_path = self.bids_list[idx][self.seg_key]

        # Load the MRI image and segmentation mask
        if self.suffix == 'fnio':
            mask = fnio.load(str(mask_path))

        elif self.suffix == 'nii.gz':
            mask = load_nifti_as_array(str(mask_path), True)

        mask = np.array(mask, dtype=np.uint8)   # Ensure mask is integer type
        mask = preprocess(mask, True, self.binary)


        #Loading multimodal stacked images
        if self.contrast == 'multimodal':
            imgs = []

            # Load all modalities
            for modality in modalities:
                img_path = self.bids_list[idx][modality]

                # Load the MRI image and segmentation mask
                if self.suffix == 'fnio':
                    img = fnio.load(str(img_path))

                elif self.suffix == 'nii.gz':
                    img = load_nifti_as_array(str(img_path))

                img = preprocess(img, False, self.binary)
                imgs.append(img)

            img = torch.cat(imgs, dim=0)    # Stack the images along the channel dimension
        

        # Load a single modality
        else:
            img_path = self.bids_list[idx][self.contrast]

            # Load the MRI image and segmentation mask
            if self.suffix == 'fnio':
                img = fnio.load(str(img_path))

            elif self.suffix == 'nii.gz':
                img = load_nifti_as_array(str(img_path))
        
            img = preprocess(img, False, self.binary)

        # Create a dictionary with the MRI image and segmentation mask
        data_dict = {self.img_key: img, self.seg_key: mask}

        if self.transform:
            data_dict = self.transform(data_dict)

        data_dict = self.postprocess(data_dict) # padding and casting to tensor

        if self.do2D:
            # Adjust as get_central_slice() expects numpy array and preprocess() returns torch tensor
            data_dict[self.img_key] = get_central_slice(data_dict[self.img_key])
            data_dict[self.seg_key] = get_central_slice(data_dict[self.seg_key])
        
        return data_dict

def create_bids_path_list_of_dicts(data_dir:str, prefix:str="", suffix:str="nii.gz")->list:
    """
    Scan the specified directory for MRI and segmentation files organized in a BIDS-like structure.

    Args:
    data_dir (str): The data directory containing subdirectories for each subject.

    prefix (str): if all folders have the same prefix e.g. "BraTS-GLI-" you can add it to be excluded from the dict key

    suffix (str): The file extension of the MRI and segmentation files. Can be modified if we saved the files in different format than .nii.gz, e.g. in fast numpy io format "fnio"

    Returns:
    dict: A dictionary where each key is a subject directory name, and the value is another
          dictionary with keys for each MRI type and the segmentation mask, containing their file paths.
    """
    list_of_path_dicts = []
    root_path = Path(data_dir)

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

def create_bids_array_list_of_dicts(data_dir:str, prefix:str= "") -> list: #  not needed anymore ?
    list_of_path_dicts = create_bids_path_list_of_dicts(data_dir, prefix)
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

def load_nifti_as_array(file_path: str, seg:bool=False)->np.ndarray:
    #Load nifti file and convert to np array
    nifti_image = NII.load(file_path, seg)
    nifti_np_array = nifti_image.get_array()
    return np.ascontiguousarray(nifti_np_array) # Ensure C-contiguity for fast numpy io

def load_normalized_central_slice_as_array(file_path: str)->np.ndarray:
    nifti_array = load_nifti_as_array(file_path)
    normalized_array = normalize(nifti_array)
    central_slice = get_central_slice(normalized_array)
    return central_slice   

def split_data_and_save_to_csv(data: list, split_percentage: tuple[float, float, float], directory: str, seed: int = 42):
    """
    Split the data into sets for training, validation, and testing and save the "subject" values to CSV files.
    Parameters:
    data (list): The list of data dictionaries to be split.
    split_percentage (tuple): The percentage of data to be used for training, validation, and testing. First element is for training, second for validation, and third for testing.
    seed (int): The seed for the random number generator.
    """
    
    # Check that the split percentages add up to 1
    if sum(split_percentage) != 1:
        raise ValueError("The sum of the split percentages must be equal to 1.")
    
    # Set the seed for reproducibility
    random.seed(seed)

    # Shuffle the data
    random.shuffle(data)
    
    train_percentage = split_percentage[0]
    val_percentage = split_percentage[1]

    # Calculate the index at which to split the data
    train_test_split_index = int(len(data) * (train_percentage + val_percentage))
    train_val_split_index = int(len(data) * train_percentage)

    # Split the data into training, validation, and testing sets
    train_data = [{'split': 'train', 'subject': d.get('subject', '')} for d in data[:train_val_split_index]]
    val_data = [{'split': 'val', 'subject': d.get('subject', '')} for d in data[train_val_split_index:train_test_split_index]]
    test_data = [{'split': 'test', 'subject': d.get('subject', '')} for d in data[train_test_split_index:]]

    # Create the directory if it doesn't exist
    Path(directory).mkdir(parents=True, exist_ok=True)

    # Save the "subject" values from the training, validation, and testing sets to CSV files
    _save_to_csv(train_data + val_data + test_data, Path(directory) / "train_val_test_split.csv")

def _save_to_csv(data: list, filename: Path):
    with filename.open('w', newline='') as csvfile:
        fieldnames = ['split', 'subject']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for d in data:
            writer.writerow(d)

def mv_files_to_folders(data: str, source_dir: str, target_dir: str):
    """
    Move files from a source directory to subdirectories in a target directory based on the subject names and the split.
    Parameters:
    data (str): Path to a CSV file containing the list of data dictionaries with the subject names.
    source_dir (str): The directory containing the directory to be moved.
    target_dir (str): The directory to which the directories will be moved.
    """
    # Load the data from the CSV file
    with open(data, mode='r') as csvfile:
        csvreader = csv.DictReader(csvfile)
        data = list(csvreader)
        
    # Convert target_dir to a Path object
    target_dir = Path(target_dir)

    # Create the target directories if they don't exist
    (target_dir / 'train').mkdir(parents=True, exist_ok=True)
    (target_dir / 'val').mkdir(parents=True, exist_ok=True)
    (target_dir / 'test').mkdir(parents=True, exist_ok=True)

    # Move the folders for the train, val and test subjects to the 'train', 'val' and 'test' directories respectively
    for folder in data:
        subject = folder.get('subject', '')
        split = folder.get('split', '')
        source_folder = Path(source_dir) / subject
        target_folder = Path(target_dir) / split / subject
        # Move the folder to the target directory
        source_folder.replace(target_folder)

def convert_nifti_to_fnio(file: str):
    # Save the NIfTI file as a fnio array in the same directory, with the same name
    nifti_np_array = load_nifti_as_array(file)
    new_file_name = file.replace('.nii.gz', '.fnio')
    fnio.save(new_file_name, nifti_np_array)

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


# class Preprocess(nn.Module):
#     """ Module to pre-process data coming from the dataset """

#     @torch.no_grad()    # disable gradients for efficiency
#     def forward(self, img, seg:bool, do2D:bool, binary:bool) -> torch.Tensor:

#         #Normalize the MRI image
#         if not seg:
#             img = normalize(img)

#         # Get a 2D slice if specified
#         if self.do2D:
#             img= get_central_slice(img)

#         # Convert segmentation mask to binary classification labels (tumor vs. non-tumor)
#         if self.binary and seg:
#             img[img > 0] = 1

#         # Convert numpy arrays to PyTorch tensors and add a channel dimension
#         if not seg: # MRI Image
#             img = torch.from_numpy(img).unsqueeze(0).float()
#         else: # Segmentation mask
#             img = torch.from_numpy(img).unsqueeze(0).long()

#         return img