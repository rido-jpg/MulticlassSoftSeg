import sys
from pathlib import Path

file = Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
sys.path.append(str(file.parents[1]))
sys.path.append(str(file.parents[2]))

from argparse import Namespace
import lightning.pytorch as pl
import numpy as np
import torch
from torchvision.datasets.cityscapes import Cityscapes
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from bids_dataset import brats_keys
from skimage.measure import block_reduce
from utils import brats_tools
from utils import cs_tools

cityscapes_root : Path = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/Cityscapes/'

class CityscapesDataModule(pl.LightningDataModule):
    def init(self, opt: Namespace = None):
        super().__init__()
        self.opt = opt
        self.batch_size = opt.bs
        self.n_workers = opt.n_cpu

    def setup(self, stage: str = None) -> None:
        self.train_dataset = CityscapesDataset(self.opt, cityscapes_root, 'train', 'fine', 'semantic')
        self.val_dataset = CityscapesDataset(self.opt, cityscapes_root, 'val', 'fine', 'semantic')
        self.test_dataset = CityscapesDataset(self.opt, cityscapes_root, 'test', 'fine', 'semantic')

    def train_dataloader(self) -> torch.Any:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=self.n_workers, pin_memory=True)
    
    def val_dataloader(self) -> torch.Any:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True)

    def test_dataloader(self) -> torch.Any:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.n_workers, pin_memory=True)

class CityscapesDataset(Dataset):
    def __init__(
        self,
        opt: Namespace,
        root: Union[str, Path],
        split: str = "train",
        mode: str = "fine",
        target_type: Union[List[str], str] = "instance",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        
        self.opt = opt
        self.root = root
        self.split = split
        self.mode = mode
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        self.ds_factor = opt.ds_factor
        self.sigma = opt.sigma

        self.dict_keys = brats_keys
        self.img_key = self.dict_keys[0]
        self.seg_key = self.dict_keys[1]
        self.soft_seg_key = self.dict_keys[2]

        self.cs_ds = Cityscapes(
            root,
            split,
            mode,
            target_type,
            transform,
            target_transform,
            transforms
        )

    def __len__(self):
        return len(self.cs_ds)
    
    def __getitem__(self, index) -> Any:
        img, gt = self.cs_ds[index]

        img_array = np.array(img) # shape (H, W, C) where C = 3 because of RGB
        reduced_gt = cs_tools.reduce_classes(gt, 20) # reduces to binary case, background is 0, traffic signs are 1; shape (H, W)

        img_array = np.transpose(img_array, (2, 0, 1)) # shape (C, H, W), where C=3 because of RGB
        reduced_gt = np.expand_dims(reduced_gt, axis=0) # shape (C, H, W), where C = 1

        if self.ds_factor is not None:
            down_img = block_reduce(img_array, block_size= (1, self.ds_factor,self.ds_factor), func = np.mean).astype(np.uint8)
            down_gt = block_reduce(reduced_gt, block_size=(1, self.ds_factor, self.ds_factor) ,func = np.mean)

            img_tensor = torch.tensor(down_img).float()
            soft_gt_tensor = torch.tensor(down_gt)
            gt_tensor = torch.round(soft_gt_tensor.clone().detach()).long()

            data_dict = {self.img_key: img_tensor, self.seg_key: gt_tensor, self.soft_seg_key: soft_gt_tensor}
            return data_dict
            
        else:
            img_tensor = torch.tensor(img_array).float()
            gt_tensor = torch.tensor(reduced_gt).long()
            soft_gt_tensor = brats_tools.soften_gt(gt_tensor.clone().detach(), self.sigma)
            data_dict = {self.img_key: img_tensor, self.seg_key: gt_tensor, self.soft_seg_key: soft_gt_tensor}
        return data_dict

