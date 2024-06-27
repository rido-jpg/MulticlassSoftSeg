import os
import sys
import torch
import TPTBox
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from pathlib import Path
from pl_unet import LitUNetModule
from TPTBox import NII
from utils.brats_tools import get_central_slice, plot_slices, preprocess
from monai.transforms import ResizeWithPadOrCrop, CastToType
from torch import nn

# file = Path(__file__).resolve()
# sys.path.append(str(file.parents[1]))
# sys.path.append(str(file.parents[2]))

# load model checkpoint
checkpoint_path = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs/3D_UNet/3D_UNet_v6_lr0.0001_batch_size_1_n_epochs_80_dimUNet_16_binary:False_with_augmentations/checkpoints/epoch=79-step=70000.ckpt'

# choose base path for image and segmentation mask
base_path = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/val/BraTS-GLI-00071-000/BraTS-GLI-00071-000-'

model_input_size = [200,200,152]
og_img_size = [240,240,155]

pad = ResizeWithPadOrCrop(og_img_size)
crop = ResizeWithPadOrCrop(model_input_size)

softmax = nn.Softmax(dim=1)

if __name__ == '__main__':
    
    # try if torch.load works, otherwise raise an error and exit
    try:
        checkpoint = torch.load(checkpoint_path)
    except Exception as e:
        print(f"Error: Invalid checkpoint path or file not found: {checkpoint_path}")
        sys.exit(1)

    # load trained model
    model = LitUNetModule.load_from_checkpoint(checkpoint_path)
    model.eval()

    # extending base path with endings for image and segmentation mask
    img_path = base_path + 't2f.nii.gz'
    seg_path = base_path + 'seg.nii.gz'

    # load mri image and segmentation mask i.e. ground truth
    try: 
        img = NII.load(img_path, seg=False)
        seg = NII.load(seg_path, seg=True)

    except Exception as e:
        print(f"Error: Invalid image or segmentation path: {img_path} or {seg_path}")
        sys.exit(1)

    # get numpy array from image and segmentation mask
    img_array = img.get_array()
    seg_array = seg.get_array()

    # normalize, turn into torch tensor, add channel dimension and ensure correct type
    img_tensor = preprocess(img_array, seg=False, binary=False)

    img_tensor = crop(img_tensor)   # crop img for model

    # add batch dimension, in this case 1 bcecause our batch size is 1
    img_tensor = img_tensor.unsqueeze(0)

    # ensure img_tensor is on same device as trained_model
    img_tensor = img_tensor.to(next(model.parameters()).device) 

    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        logits = model(img_tensor)  # get logits from model
        probs = softmax(logits) # apply softmax to get probabilities
        preds = torch.argmax(probs, dim=1) # get class with highest probability
        preds_cpu = preds.cpu() # move tensor to cpu
        del logits, probs, preds # delete tensors to free up memory

    preds_cpu = preds_cpu.type(torch.float16)    # change type of preds_cpu from torch.int64 to torch.float16

    preds_padded = pad(preds_cpu) # pad preds_cpu to original image size
    preds_padded = preds_padded.squeeze(0) # get rid of batch dimension

    preds_array = preds_padded.numpy() # convert to numpy array

    preds_array = preds_array.astype(np.uint16) # ensure correct type

    pred_nii: NII = seg.set_array(preds_array)  # get prediction as nii object

    pred_nii.save(base_path + "pred.nii.gz")    # save prediction as nifti file to view it in ITK Snap

    # get difference between original segmentation mask and prediction
    difference_nifti = NII.get_segmentation_difference_to(pred_nii, seg, ignore_background_tp=True)
    difference_nifti.save(base_path + "seg-difference.nii.gz")