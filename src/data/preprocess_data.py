from bids_dataset import create_bids_path_list_of_dicts, save_bids_niftis_as_fnio
from pathlib import Path

# Preprocess Nifti images used for training, validation and testing
# A script that is only used once, which loops over the whole BIDS folder structure, and saves all files in fnio (fast numpy io) format in the same folders with the same name (just the file ending is replaced by .fnio).

bids_root_path = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
list_of_path_dicts = create_bids_path_list_of_dicts(bids_root_path)

save_bids_niftis_as_fnio(list_of_path_dicts)