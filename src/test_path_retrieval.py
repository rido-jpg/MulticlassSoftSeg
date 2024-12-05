# %%
from data.bids_dataset import  BidsDataset
from pathlib import Path
import re
from pl_unet import LitUNetModule
# %%

def _get_gt_paths(gt_dir : Path) -> list:
    gt_dir : Path = Path(gt_dir)
    gt_paths = list(gt_dir.rglob("*seg.nii.gz"))
    return gt_paths

def _extract_brats_id(filename):
    # Define the regular expression pattern to match BraTS-GLI-XXXXXX-XXX
    pattern = r"(BraTS-GLI-\d{5}-\d{3})"
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)  # Return the matched part
    else:
        return None  # Return None if no match is found

def get_best_checkpoint(directory: str):
    # Define the directory path
    directory_path = Path(directory) / "checkpoints"
    
    # Regular expression pattern to capture val_diceFG value
    pattern = re.compile(r"val_diceFG=([0-9.]+)")

    best_ckpt = None
    best_val_diceFG = -1

    # Iterate over all .ckpt files in the checkpoints folder
    for ckpt_file in directory_path.glob("*.ckpt"):
        match = pattern.search(ckpt_file.name)
        if match:
            # Convert the captured value to a float
            val_diceFG = float(match.group(1))
            # Check if this is the highest val_diceFG so far
            if val_diceFG > best_val_diceFG:
                best_val_diceFG = val_diceFG
                best_ckpt = ckpt_file

    return best_ckpt

# %%
if __name__ == '__main__':
    data_dir = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test"
    baseline_exp1_path = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs/brats/exp_1/3D_UNet/3D_UNet_v0_lr0.0001_ce_1.0_hard_1.0_soft_1.0_soft_dice_1.0_sigma_0.125_softmax_Baseline_exp_1")
    baseline_exp1_ckpt_path = get_best_checkpoint(baseline_exp1_path)
    hparams_baseline_exp1 = LitUNetModule.load_from_checkpoint(baseline_exp1_ckpt_path).hparams
    baseline_exp1_opt = hparams_baseline_exp1.get('opt')

    bids_baseline_exp1_ds = BidsDataset(baseline_exp1_opt, data_dir)

    og_hard_gt_list = _get_gt_paths(data_dir)

    for idx, subject in enumerate(zip(bids_baseline_exp1_ds, og_hard_gt_list)):
            ds_dict = subject[0]
            dict_subject = bids_baseline_exp1_ds.bids_list[idx]['subject']

            list_path = subject[1]
            list_subject = _extract_brats_id(str(list_path))

            print(f"subject from dict: {dict_subject} and subject from list {list_subject}")
            assert dict_subject == list_subject, "The subjects do not align"

            if idx > 10:
                break


# %%
