# %%
import subprocess
from pathlib import Path
import re

def _extract_unet_version(path: Path) -> str:
    # Convert Path object to string
    path_str = str(path)
    
    # Regular expression to match '3D_UNet_v' followed by a number
    match = re.search(r"3D_UNet_v\d+", path_str)
    
    # Return the matched part if found, otherwise return None
    return match.group(0) if match else None

if __name__ == "__main__":

    ## INFERENCE PART

    python_interpreter =Path("/home/student/farid_ma/dev/multiclass_softseg/bin/python") 

    inference_path= Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/inference_v2.py")

    model_path= Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/logs/lightning_logs/brats/exp_1/3D_UNet/3D_UNet_v0_lr0.0001_ce_1.0_hard_1.0_soft_1.0_soft_dice_1.0_sigma_0.125_softmax_Baseline_exp_1")

    data_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test")

    suffix = "_exp1_baseline_test_set"

    activation = "softmax"

    inf_args = ["-model_dir",str(model_path),"-niftis","-soft","-slices","-activation",activation, "-suffix",suffix, "-data_dir", str(data_dir)]

    # Run the script with the environment-specific Python interpreter
    result = subprocess.run([python_interpreter, inference_path] + inf_args, capture_output=True, text=True)

    # Print the output and any errors
    print("Output:", result.stdout)
    print("Errors:", result.stderr)
    # %%
    ## EVAL PART
    eval_path = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/panoptica_eval.py")

    model_name = f"{_extract_unet_version(model_path)}_{suffix}"
    preds_base_path = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/preds")
    model_preds_path = preds_base_path / model_name 

    eval_args = ["-preds_dir", str(model_preds_path)]

    # Run the script with the environment-specific Python interpreter
    result = subprocess.run([python_interpreter, eval_path] + eval_args, capture_output=True, text=True)

    # Print the output and any errors
    print("Output:", result.stdout)
    print("Errors:", result.stderr)
