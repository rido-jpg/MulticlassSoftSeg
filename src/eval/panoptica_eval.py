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

dir = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval"

prefix = "BraTS-GLI-"
subject_id = "00429-000"
subject = f"{prefix}{subject_id}"
data_dir = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/"

evaluator = Panoptica_Aggregator(
    Panoptica_Evaluator.load_from_config_name("panoptica_evaluator_BRATS"),
    f"{dir}/test.tsv",
    log_times=True,
)

evaluator.panoptica_evaluator.set_log_group_times(True)

if __name__ == "__main__":
    ref_mask = NII.load(f"{data_dir}/val/{subject}/{subject}-seg.nii.gz", True).get_array()
    pred_mask = NII.load(f"{data_dir}/preds/{subject}/{subject}-pred-HARD_DiceFG_0.878_relu.nii.gz", True).get_array()
    #pred_mask = NII.load(f"{data_dir}/preds/{subject}/{subject}-pred-MIXED_MSE_DiceFG_0.86_relu.nii.gz", True).get_array()

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(
                evaluator.evaluate, pred_mask, ref_mask, f"sample{i}"
            )
            for i in range(10)
        }
        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Panoptica Evaluation"
        ):
            result = future.result()
            if result is not None:
                print("Done")

    panoptic_statistic = evaluator.make_statistic()
    panoptic_statistic.print_summary()

    fig = panoptic_statistic.get_summary_figure("sq_dsc", horizontal=True)
    out_figure = str(f"{dir}/example_sq_dsc_figure.png")
    fig.write_image(out_figure)

    fig2 = make_curve_over_setups(
        {
            "t0.5": panoptic_statistic,
            "bad": panoptic_statistic,
            "good classifier": panoptic_statistic,
            2.0: panoptic_statistic,
        },
        groups=None,
        metric="pq",
    )

    out_figure = str(f"{dir}/example_multiple_statistics.png")
    fig2.savefig(out_figure)

