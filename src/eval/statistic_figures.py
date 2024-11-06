# %%
from panoptica.panoptica_statistics import Panoptica_Statistic, make_curve_over_setups
from pathlib import Path

eval_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/figures")

baseline_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/3D_UNet_v0__Hard_Baseline.tsv"
v1_mixed_loss_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/3D_UNet_v1__exp_1_mixed_loss.tsv"
v2_mse_relu_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/3D_UNet_v2__exp1_mse_relu.tsv"
v2_mse_softmax_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/3D_UNet_v2__exp1_mse_softmax.tsv"

baseline = Panoptica_Statistic.from_file(baseline_path)
v1_mixed_loss = Panoptica_Statistic.from_file(v1_mixed_loss_path)
v2_mse_relu = Panoptica_Statistic.from_file(v2_mse_relu_path)
v2_mse_softmax = Panoptica_Statistic.from_file(v2_mse_softmax_path)

# fig = model_1.get_summary_figure("sq_dsc", horizontal=True)
# out_figure = str(f"{eval_dir}/example_sq_dsc_figure.png")
# fig.write_image(out_figure)

fig2 = make_curve_over_setups(
    {
        "baseline": baseline,
        "v1_mixed_loss": v1_mixed_loss,
        "v2_mse_relu": v2_mse_relu,
        "v2_mse_softmax": v2_mse_softmax,

    },
    groups='wt (1, 2, 3)',
    metric="global_bin_dsc",
)

out_figure = str(f"{eval_dir}/wt-global_bin_dsc.png")
fig2.savefig(out_figure)