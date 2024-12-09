# %%
import sys
from pathlib import Path
file = Path(__file__).resolve()
sys.path.append(str(file.parents[0]))
sys.path.append(str(file.parents[1]))
# %%

# %%
from skimage.measure import block_reduce
import numpy as np
import torch.nn.functional as F
from torch import from_numpy
from TPTBox import NII
# %%
from bids_dataset import create_bids_path_list_of_dicts, contrasts, load_nifti_as_array, soften_gt
#from utils.brats_tools import plot_slices, get_central_slice

# %%
def _get_gt_paths(gt_dir : Path) -> list:
    gt_dir : Path = Path(gt_dir)
    gt_paths = list(gt_dir.rglob("*seg.nii.gz"))
    return gt_paths

if __name__ == '__main__':
    #sigma = 0.125
    sigma = 0.5
    if sigma == 0.125:
        sig_sfx = "low"
    elif sigma == 0.5:
        sig_sfx = "high"
    else:
        sys.exit("Sigma has to be 0.125 or 0.5")
    # %%
    soft_ds_gt_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test_ds")
    target_dir = Path(soft_ds_gt_dir).parent.joinpath(f'test_sig_{sig_sfx}')
    # %%

    gt_paths = _get_gt_paths(soft_ds_gt_dir)
    sample_nifti = gt_paths[0]

    # %%
    for soft_gt_path in gt_paths:
        sample_nii = NII.load(sample_nifti, seg = False)    # random sample we just use as "skeleton"
        sample_nii.set_dtype_(np.float32)

        new_path = Path(str(soft_gt_path).replace("test_ds", f"test_sig_{sig_sfx}"))   # Replace "test_ds" with "test_sig_low" in the path
        new_path = new_path.parent / new_path.name.replace("ds-seg.nii.gz", f"ds-rebin-{sig_sfx}-sig-seg.nii.gz")  # Replace the last part of the file name
        print(f"Current file path: {new_path}")

        if soft_gt_path and Path(soft_gt_path).exists():  # Ensure the file exists
            ds_foreground_nifti = NII.load(soft_gt_path, seg=False).get_array() # only foreground channel
            rebin_arr = (ds_foreground_nifti >= 0.5).astype(np.uint8)    # binarize by rounding
            rebin_tensor = from_numpy(rebin_arr).unsqueeze(0)  # turn into tensor and add channel to get shape (C, H, W, D)
            soft_nifti = soften_gt(rebin_tensor, sigma = sigma).squeeze(0).numpy()   # soften and turn into np array
            soft_nii : NII = sample_nii.set_array(soft_nifti)
        soft_nii.save(new_path)
    #%%

