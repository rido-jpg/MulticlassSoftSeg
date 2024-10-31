import argparse
import sys
import re

from panoptica import (
    Panoptica_Evaluator,
    Panoptica_Aggregator,
    InputType,
    NaiveThresholdMatching,
    ConnectedComponentsInstanceApproximator,
    Metric,
)

#from panoptica.utils import SegmentationClassGroups, LabelGroup
from panoptica.panoptica_statistics import make_curve_over_setups
from pathlib import Path
from panoptica.utils import NonDaemonicPool
from joblib import delayed, Parallel
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from multiprocessing import set_start_method

from TPTBox import NII

def parse_inf_param(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser()
    
    parser.add_argument("-preds_dir", type=str, default = None, help="Path to the model predictions")
    parser.add_argument("-gt_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/val", help="Path to the Ground Truths (BIDS format)")
    parser.add_argument("-save_dir", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval", help="Path to the directory where the evaluation .tsv should be saved")
    parser.add_argument("-test_loop", action='store_true', help="Run test loop with single sample for debugging")
    parser.add_argument("-samples", type=int, default = 0, help="Number of samples to evaluate. If 0, evaluate all predictions")
    parser.add_argument("-config", type=str, default = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/panoptica_evaluator_BRATS.yaml", help="Name of the configuration file to use for the evaluation")

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

def _get_gt_paths(gt_dir : Path) -> list:
    gt_dir : Path = Path(gt_dir)
    gt_paths = list(gt_dir.rglob("*seg.nii.gz"))
    return gt_paths

def _proc(evaluator, pred: Path, gt: Path):
    pred_arr = NII.load(pred, True).get_seg_array()
    gt_arr = NII.load(gt, True).get_seg_array()
    if _extract_brats_id(str(pred)) != _extract_brats_id(str(gt)):
        # throw assertion error
        raise Exception(f"The prediction and ground truth do not match. Prediction subject: {_extract_brats_id(str(pred))}, Ground truth subject: {_extract_brats_id(str(gt))}")
    evaluator.evaluate(pred_arr, gt_arr,_extract_brats_id(str(gt)))

def _extract_brats_id(filename):
    # Define the regular expression pattern to match BraTS-GLI-XXXXXX-XXX
    pattern = r"(BraTS-GLI-\d{5}-\d{3})"
    
    # Search for the pattern in the filename
    match = re.search(pattern, filename)
    
    if match:
        return match.group(1)  # Return the matched part
    else:
        return None  # Return None if no match is found
    
def _extract_unet_version(path: Path) -> str:
    # Convert Path object to string
    path_str = str(path)
    
    # Regular expression to match '3D_UNet_v' followed by a number
    match = re.search(r"3D_UNet_v\d+", path_str)
    
    # Return the matched part if found, otherwise return None
    return match.group(0) if match else None

# prefix = "BraTS-GLI-"
# subject_id = "00429-000"
# subject = f"{prefix}{subject_id}"
# data_dir = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/"

if __name__ == "__main__":
    parser = parse_inf_param()
    conf = parser.parse_args()

    preds_dir : Path = Path(conf.preds_dir)
    gt_dir : Path = Path(conf.gt_dir)
    save_dir : Path = Path(conf.save_dir)
    config = conf.config
    samples = conf.samples

    if conf.test_loop:
        samples = 1

    model_name = preds_dir.stem

    evaluator = Panoptica_Aggregator(
        Panoptica_Evaluator.load_from_config(config),
        f"{save_dir}/{model_name}.tsv",
        log_times=True,
    )

    evaluator.panoptica_evaluator.set_log_group_times(True)

    pred_paths = _get_pred_paths(preds_dir)
    gt_paths = _get_gt_paths(gt_dir)

    if len(pred_paths) != len(gt_paths):
        print("The number of predictions and ground truths do not match")
        sys.exit()

    if samples > 0 and samples < len(pred_paths):
        iterations = samples
    else:
       iterations = len(pred_paths)

    with ProcessPoolExecutor() as executor:
        futures = {executor.submit( _proc, evaluator, pred_paths[idx], gt_paths[idx]) for idx in range(iterations)}
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Panoptica Evaluation"
        ):
            result = future.result()
            if result is not None:
                print("Done")

    panoptic_statistic = evaluator.make_statistic()
    panoptic_statistic.print_summary()

    # fig = panoptic_statistic.get_summary_figure("sq_dsc", horizontal=True)
    # out_figure = str(f"{dir}/example_sq_dsc_figure.png")
    # fig.write_image(out_figure)

    # fig2 = make_curve_over_setups(
    #     {
    #         "t0.5": panoptic_statistic,
    #         "bad": panoptic_statistic,
    #         "good classifier": panoptic_statistic,
    #         2.0: panoptic_statistic,
    #     },
    #     groups=None,
    #     metric="pq",
    # )

    # out_figure = str(f"{dir}/example_multiple_statistics.png")
    # fig2.savefig(out_figure)

