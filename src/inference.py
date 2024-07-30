import os
import sys
import torch
import TPTBox
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import utils.fastnumpyio.fastnumpyio as fnio
import TPTBox
from pathlib import Path
from argparse import Namespace
from pl_unet import LitUNetModule
from TPTBox import NII
from utils.brats_tools import get_central_slice, plot_slices
from monai.transforms import ResizeWithPadOrCrop, CastToType
from torch import nn
from data.bids_dataset import modalities, create_bids_path_list_of_dicts, BidsDataset

# file = Path(__file__).resolve()
# sys.path.append(str(file.parents[1]))
# sys.path.append(str(file.parents[2]))

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


### RUN SPECIFIC CONFIGURATION ###
checkpoint_path = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs/3D_UNet/3D_UNet_v11_lr0.0001_batch_size_1_n_epochs_400_dimUNet_16_binary:False_with_augmentations_/checkpoints/epoch=372-step=81687-val_diceFG=0.8407_valdiceFG-weights.ckpt'
data_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/val'
suffix = "3D_UNet_v11"  # suffix for saved files
save_gts = False        # GTs should only need to be saved once
test_loop = True        # test loop with single sample for debugging
format = 'fnio'         # format of the data


# load hparams from checkpoint to get contrast
hparams = LitUNetModule.load_from_checkpoint(checkpoint_path).hparams
conf = hparams.get('conf')
contrast = get_option(conf, 'contrast', 't2f')  # if the contrast is not part of hparams, it is an old ckpt which used 't2f'


model_input_size = get_option(conf, 'resize', [200, 200, 152])

print(f"Model input size: {model_input_size}")

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

    # create BidsDataset instance
    bids_val_ds = BidsDataset(data_dir, contrast=contrast, suffix=format, resize=model_input_size)

    for idx, dicts in enumerate(bids_val_ds):
        img = dicts['img']

        subject = bids_val_ds.bids_list[idx]['subject']

        print(f"---------------------------------")
        print(f"Subject: {subject}")

        base_path = data_dir + '/' + subject + '/' + subject

        # add batch dimension, in this case 1 because our batch size is 1
        img_tensor = img.unsqueeze(0)

        # ensure img_tensor is on same device as trained_model
        img_tensor = img_tensor.to(next(model.parameters()).device) 

        with torch.no_grad():
            logits = model(img_tensor)  # get logits from model
            probs = softmax(logits) # apply softmax to get probabilities
            preds = torch.argmax(probs, dim=1) # get class with highest probability
            preds_cpu = preds.cpu() # move tensor to cpu
            del logits, probs, preds # delete tensors to free up memory

        preds_cpu = preds_cpu.type(torch.float16)    # change type of preds_cpu from torch.int64 to torch.float16

        preds_padded = pad(preds_cpu) # pad preds_cpu to original image size
        preds_padded = preds_padded.squeeze(0) # get rid of batch dimension

        preds_array = preds_padded.numpy() # convert to numpy array (dtype: float16)

        preds_array = preds_array.astype(np.uint8) # ensure correct type to be able to cast it to a NII object

        # load GT segmentation as NII object 
        seg_path = base_path + '-seg.nii.gz'
        seg = NII.load(seg_path, seg=True)

        pred_nii: NII = seg.set_array(preds_array)  # get prediction as nii object

        pred_nii.save(base_path + "-pred-" + suffix + ".nii.gz")    # save prediction as nifti file to view it in ITK Snap

        # get difference between original segmentation mask and prediction
        difference_nifti = NII.get_segmentation_difference_to(pred_nii, seg, ignore_background_tp=True)
        difference_nifti.save(base_path + "-seg-difference-" + suffix + ".nii.gz")

        if save_gts:
            gt_slice = get_central_slice(seg.get_array()) # get central slice of ground truth

        pred_slice = get_central_slice(preds_array) # get central slice of prediction

        for contrast in modalities:
            if format == 'nii.gz':
                full_img = NII.load(bids_val_ds.bids_list[idx][contrast], seg=False)
                img_array = full_img.get_array()
            elif format == 'fnio':
                img_array = fnio.load(str(bids_val_ds.bids_list[idx][contrast]))

            img_slice = get_central_slice(img_array)
            plot_slices(img_slice, pred_slice,plt_title='Prediction '+ suffix , save_path=base_path + f"-{contrast}-slice-pred-{suffix}.png", show=False)
            if save_gts:
                plot_slices(img_slice, gt_slice, plt_title='Ground Truth',save_path=base_path + f"-{contrast}-slice-gt.png", show = False)

        if test_loop:     
            # for testing purposes -> exit after one iteration
            break
