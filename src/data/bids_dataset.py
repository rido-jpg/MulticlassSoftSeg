import os
import logging
#import nibabel as nib
import random
import csv
import sys
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import torch
import torch.nn.functional as F
import numpy as np
import lightning.pytorch as pl

from argparse import Namespace
from TPTBox import NII
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from skimage.measure import block_reduce

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
from utils.brats_tools import preprocess, slice_and_pad, normalize, get_central_slice, soften_gt, brats_load, load_nifti_as_array

# Configure logging
logging.basicConfig(level=logging.INFO)

# Array of all MRI contrasts and the segmentation mask  
contrasts = ['t1c', 't1n', 't2f', 't2w', 'seg']
modalities = ['t1c', 't1n', 't2f', 't2w']

# Keys of Dictionary that will be returned by the Dataset
brats_keys = ['img', 'seg', 'soft_seg']

# Dictionary with each brats region and its corresponding labels
brats_regions = {'ET': [3], 'TC': [1, 3], 'WT': [1, 2, 3]}

# # Train Transforms
# train_transforms = Compose(
#     [
#         RandRotateD(keys=brats_keys, range_x=[- math.radians(30), math.radians(30)], prob=1, mode =["bilinear", "nearest"]),
#     ]
# )

class BidsDataModule(pl.LightningDataModule):
    def __init__(self, opt: Namespace, data_dir: str, format: str='fnio', train_transform=None, test_transform=None):
        super().__init__()
        self.opt = opt
        self.data_dir = data_dir
        self.format = format
        self.batch_size = opt.bs 
        self.train_transform = train_transform   
        self.test_transform = test_transform
        self.n_workers = opt.n_cpu

    def setup(self, stage: str = None) -> None:
        self.train_dataset = BidsDataset(self.opt, self.data_dir +'/train', suffix=self.format, transform=self.train_transform)
        self.val_dataset = BidsDataset(self.opt, self.data_dir + '/val', suffix=self.format, transform=self.test_transform)
        self.test_dataset = BidsDataset(self.opt, self.data_dir + '/test', suffix=self.format, transform=self.test_transform)
        
    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.n_workers, pin_memory=True)
    
    def val_dataloader(self) -> torch.Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    def test_dataloader(self) -> torch.Any:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True)

class BidsDataset(Dataset):
    def __init__(self, opt: Namespace, data_dir:str, prefix:str="", suffix:str='fnio', transform=None):
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
        self.opt = opt
        self.data_dir = data_dir   
        self.prefix = prefix     
        self.contrast = opt.contrast
        self.suffix = suffix
        self.do2D = opt.do2D
        self.transform = transform
        self.binary = opt.binary
        #self.soft = opt.soft
        #self.one_hot = opt.one_hot

        self.ds_factor = opt.ds_factor
        self.sigma = opt.sigma

        self.resize = tuple(opt.resize)

        self.dict_keys = brats_keys
        self.img_key = self.dict_keys[0]
        self.seg_key = self.dict_keys[1]
        self.soft_seg_key = self.dict_keys[2]

        if self.binary:
            self.n_classes = 2
        else:
            self.n_classes = 4

        if self.do2D:
            self.resize = self.resize[:2]

        self.postprocess = Compose(
            [
                ResizeWithPadOrCropd(keys=brats_keys, spatial_size=self.resize, mode="symmetric"),
                CastToTyped(keys=brats_keys, dtype=(torch.float, torch.long, torch.float)),
            ]
        )

        self.bids_list = create_bids_path_list_of_dicts(self.data_dir, prefix=self.prefix, suffix=self.suffix)
        
    def __len__(self):
        return len(self.bids_list)
    
    def __getitem__(self, idx):
        """"
        returns: dict with keys 'img' and 'seg' containing the MRI image and segmentation mask
        """
        mask_path = self.bids_list[idx][self.seg_key]   # Get the path to the segmentation mask
        mask = brats_load(str(mask_path), self.suffix, True)    # Load the segmentation mask

        hard_mask = preprocess(mask, seg=True, binary=self.binary, opt=self.opt)
        if self.ds_factor is None:
            soft_mask = preprocess(mask, seg=True, binary=self.binary, n_classes=self.n_classes, opt=self.opt, soft=True)
        else: 
            soft_mask = torch.zeros_like(hard_mask)

        #Loading multimodal stacked images
        if self.contrast == 'multimodal':
            imgs = []

            # Load all modalities
            for modality in modalities:
                img_path = self.bids_list[idx][modality]
                img = brats_load(str(img_path), self.suffix)    # Load the MRI image
                img = preprocess(img, seg=False, binary=self.binary)    # Preprocess the MRI image
                imgs.append(img)

            img = torch.cat(imgs, dim=0)    # Stack the images along the channel dimension
        

        # Load a single modality
        else:
            img_path = self.bids_list[idx][self.contrast]
            img = brats_load(str(img_path), self.suffix)    # Load the MRI image
            img = preprocess(img, seg=False, binary=self.binary)    # Preprocess the MRI image

        # Create a dictionary with the MRI image and segmentation mask
        data_dict = {self.img_key: img, self.seg_key: hard_mask, self.soft_seg_key: soft_mask}

        if self.transform:
            data_dict = self.transform(data_dict)

        data_dict = self.postprocess(data_dict) # padding and casting to tensor

        if self.ds_factor is not None:
            oh_gt = F.one_hot(data_dict[self.seg_key].squeeze(0), num_classes=self.n_classes).permute(3,0,1,2)
            down_img = block_reduce(np.array(data_dict[self.img_key]), block_size= (1, self.ds_factor, self.ds_factor, self.ds_factor), func = np.mean)
            down_gt = block_reduce(np.array(oh_gt), block_size=(1, self.ds_factor, self.ds_factor, self.ds_factor) ,func = np.mean)

            img_tensor = torch.tensor(down_img).float()
            soft_gt_tensor = torch.tensor(down_gt).float()
            gt_tensor = torch.argmax(soft_gt_tensor, dim=0).unsqueeze(0)    # rebinarizing the soft GT to a hard GT

            data_dict = {self.img_key: img_tensor, self.seg_key: gt_tensor, self.soft_seg_key: soft_gt_tensor}
            pass


        if self.do2D:
            # Adjust as get_central_slice() expects numpy array and preprocess() returns torch tensor
            data_dict[self.img_key] = get_central_slice(data_dict[self.img_key])
            data_dict[self.seg_key] = get_central_slice(data_dict[self.seg_key])
            data_dict[self.soft_seg_key] = get_central_slice(data_dict[self.soft_seg_key])
        
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
