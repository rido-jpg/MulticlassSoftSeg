# %%
import torchmetrics.functional as mF
import torch
import argparse
import sys
from pathlib import Path
from TPTBox import NII
import numpy as np

def parse_inf_param(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument("-preds_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/preds/3D_UNet_v0__exp1_baseline_test_set", help="Path to the model predictions")
    parser.add_argument("-gt_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/test", help="Path to the Ground Truths (BIDS format)")
    parser.add_argument("-save_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics", help="Path to the directory where the evaluation .tsv should be saved")

    return parser

def _get_pred_paths(preds_dir : Path) -> list:
    """"
    Get the paths of the predictions in the preds_dir
    
    """
    preds_dir : Path = Path(preds_dir)
    pred_paths = []

    for file_path in preds_dir.rglob("*.nii.gz"):
        if "pred" in file_path.name:
            pred_paths.append(file_path)

    return pred_paths

def _get_channelwise_niftis_folders(preds_dir: Path) -> list:
    preds_dir = Path(preds_dir)
    niftis_paths = list(preds_dir.rglob("niftis"))
    return niftis_paths

def _get_gt_paths(gt_dir : Path) -> list:
    gt_dir : Path = Path(gt_dir)
    gt_paths = list(gt_dir.rglob("*seg.nii.gz"))
    return gt_paths

# h, w, d = opt.resize

# if ds_factor is not None:
#     h, w, d = [x//ds_factor for x in (h, w, d)]

def soft_metrics(soft_scores: torch.Tensor, preds: torch.Tensor, gt: torch.Tensor, soft_gt:torch.Tensor, img_area:int):
    """
    Calculate regression metrics for predictions, ground truths, soft predictions/scores and soft gt

    Params:
    soft_scores (torch.Tensor): soft sample-wise outputs of model with shape [C, H, W, D]
    preds (torch.Tensor): sample-wise predictions of model with shape [H, W, D]
    gt (torch.Tensor): sample-wise gt of model with shape [H, W, D]
    soft_gt (torch.Tensor): soft sample-wise gts of model with shape [C, H, W, D]

    returns:
    mae (float)
    mse (float)
    fmse (float)
    soft_mse (float)
    """

    mae = mF.mean_absolute_error(preds, gt)

    mse = mF.mean_squared_error(preds,gt)

    fore_mask = (gt>0).long()
    fore_area = torch.sum(fore_mask).item()

    img_area = preds.view(-1).shape[0]

    if fore_area > 0:  # Avoid division by zero
        fmse = mF.mean_squared_error(preds * fore_mask, gt * fore_mask) * img_area / fore_area
    else:
        fmse = torch.tensor(0.0)  # Handle the case where there is no foreground

    #soft_mse = mF.mean_squared_error(soft_scores, soft_gt)
    return mae, mse, fmse
    # return mae, mse, fmse, soft_mse

# %%

if __name__ == '__main__':
    parser = parse_inf_param()
    conf = parser.parse_args()

    preds_dir : Path = Path(conf.preds_dir)
    gt_dir : Path = Path(conf.gt_dir)

    pred_paths = _get_pred_paths(preds_dir)
    gt_paths = _get_gt_paths(gt_dir)

    if len(pred_paths) != len(gt_paths):
        print("The number of predictions and ground truths do not match")
        sys.exit()

    soft_score_parent_dir_paths = _get_channelwise_niftis_folders(preds_dir)

    for pred_path, gt_path in zip(pred_paths, gt_paths):
        subject = gt_path.parent.name

        for part in pred_path.parts:
            if "BraTS-GLI-" in part:  # Check if the part contains the desired keyword
                if subject != part:
                    print(f"The prediction list is at subject {part} while the gt list is at subject {subject}")
                    sys.exit()
                break

        pred = NII.load(pred_path, seg = False)
        gt = NII.load(gt_path, seg = True)

        pred_array = pred.get_array()
        gt_array = gt.get_array().astype(dtype = np.uint8)

        pred_tensor = torch.from_numpy(pred_array)
        gt_tensor = torch.from_numpy(gt_array)

        print(soft_metrics(None, pred_tensor, gt_tensor, None, None))
        break



# %%
