# %%
from panoptica.panoptica_statistics import Panoptica_Statistic, make_curve_over_setups
from pathlib import Path

eval_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval")

baseline = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/3D_UNet_v0__Hard_Baseline.tsv"
path_1 = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/3D_UNet_v2__exp1_mse_relu.tsv"
path_2 = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/3D_UNet_v2__exp1_mse_softmax.tsv"

baseline = Panoptica_Statistic.from_file(baseline)
model_1 = Panoptica_Statistic.from_file(path_1)
model_2 = Panoptica_Statistic.from_file(path_2)

fig = model_1.get_summary_figure("sq_dsc", horizontal=True)
out_figure = str(f"{eval_dir}/example_sq_dsc_figure.png")
fig.write_image(out_figure)

fig2 = make_curve_over_setups(
    {
        "mse_relu": model_1,
        "mse_softmax": model_2,
        "baseline": baseline,
    },
    groups=None,
    metric="sq_dsc",
)

out_figure = str(f"{eval_dir}/example_multiple_statistics.png")
fig2.savefig(out_figure)