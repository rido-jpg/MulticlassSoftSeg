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
from bids_dataset import create_bids_path_list_of_dicts, contrasts, load_nifti_as_array
#from utils.brats_tools import plot_slices, get_central_slice
# %%
def downsample(nifti_path:Path, seg:bool):
    """
    Assumes nifti to be of shape [H, W, D]
    """
    nifti = load_nifti_as_array(nifti_path, seg)
    if seg:
        nifti = (nifti > 0) #binarize
        oh_nifti = from_numpy(nifti).long()
        oh_nifti = F.one_hot(oh_nifti, num_classes=2).permute(3,0,1,2) # shape (2, H, W, D)
        return block_reduce(np.array(oh_nifti), (1,2,2,2), func = np.mean)
    else: 
        return block_reduce(nifti, func = np.mean)
# %%
bids_test_path = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test'
list_of_path_dicts = create_bids_path_list_of_dicts(bids_test_path)
target_dir = Path(bids_test_path).parent.joinpath('test_ds')
# %%
sample_dict = list_of_path_dicts[0]
sample_nifti = sample_dict['t2f']
sample_seg = sample_dict['seg']
sample_nii = NII.load(sample_nifti, seg = False)
sample_nii.set_dtype_(np.float32)
# %%
for subject_dict in list_of_path_dicts:
    for contrast in contrasts:
        sample_nii = NII.load(sample_nifti, seg = False)
        sample_nii.set_dtype_(np.float32)

        filepath = subject_dict.get(contrast)
        old_sfx = f"{contrast}.nii.gz"
        new_sfx = f"ds-{contrast}.nii.gz"
        new_name = filepath.name.replace(old_sfx,new_sfx)
        new_path = target_dir.joinpath(subject_dict.get('subject')).joinpath(new_name)
        print(f"subject: {subject_dict.get('subject')} and contrast: {contrast}")
        if contrast == 'seg':
            seg = True
        else:
            seg = False
        if filepath and Path(filepath).exists():  # Ensure the file exists
            ds_nifti = downsample(Path(filepath), seg)
            if seg:
                ds_nii : NII = sample_nii.set_array(ds_nifti[1])
            else:
                ds_nii : NII = sample_nii.set_array(ds_nifti)
        print(ds_nii)

        ds_nii.save(new_path)
 #%%

