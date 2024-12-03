# %%
from panoptica.panoptica_statistics import Panoptica_Statistic, make_curve_over_setups
from pathlib import Path
import pandas as pd

output_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics")

baseline_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v0/3D_UNet_v0__exp1_baseline_test_set.tsv"
baseline_processed_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v0/3D_UNet_v0__baseline_test_set_postprocessed.tsv"
mse_low_sig_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v2_mse_low_sigma/3D_UNet_v2__exp_1_mse_lowest_sigma_linear_test_set.tsv"
mse_low_sig_processed_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v2_mse_low_sigma/3D_UNet_v2__exp_1_mse_lowest_sigma_linear_test_set_postprocessed.tsv"
mse_high_sig_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v11_mse_high_sigma/3D_UNet_v11__exp1_mse_high_sigma_linear_test_set.tsv"
mse_high_sig_processed_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v11_mse_high_sigma/3D_UNet_v11__exp_1_mse_highest_sigma_linear_test_set_postprocessed.tsv"
adw_low_sig_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v5_adw_low_sigma/3D_UNet_v5__exp1_adw_lowest_sigma_linear_test_set.tsv"
adw_low_sig_processed_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v5_adw_low_sigma/3D_UNet_v5__exp1_adw_lowest_sigma_linear_test_set_postprocessed.tsv"
#adw_mid_sig_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/3D_UNet_v6__exp1_adw_mid_sigma_linear_test_set.tsv"
adw_high_sig_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v10_adw_high_sigma/3D_UNet_v10__exp1_adw_highest_sigma_linear_test_set.tsv"
adw_high_sig_processed_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/exp1/test_set/v10_adw_high_sigma/3D_UNet_v10__exp1_adw_highest_sigma_linear_test_set_postprocessed.tsv"

baseline = Panoptica_Statistic.from_file(baseline_path)
baseline_processed = Panoptica_Statistic.from_file(baseline_processed_path)
mse_low_sig = Panoptica_Statistic.from_file(mse_low_sig_path)
mse_low_sig_processed = Panoptica_Statistic.from_file(mse_low_sig_processed_path)
mse_high_sig = Panoptica_Statistic.from_file(mse_high_sig_path)
mse_high_sig_processed = Panoptica_Statistic.from_file(mse_high_sig_processed_path)
adw_low_sig = Panoptica_Statistic.from_file(adw_low_sig_path)
adw_low_sig_processed = Panoptica_Statistic.from_file(adw_low_sig_processed_path)
#adw_mid_sig = Panoptica_Statistic.from_file(adw_mid_sig_path)
adw_high_sig = Panoptica_Statistic.from_file(adw_high_sig_path)
adw_high_sig_processed = Panoptica_Statistic.from_file(adw_high_sig_processed_path)

# # %% 
# data = model_2.get_summary_dict()

# # Transform the data into a format suitable for CSV
# rows = []
# for category, metrics in data.items():
#     for metric, (avg, std) in metrics.items():
#         rows.append({
#             'Category': category,
#             'Metric': metric,
#             'Average': avg,
#             'StdDev': std
#         })

# # Create a DataFrame and save to CSV
# df = pd.DataFrame(rows)
# model_name = Path(path_2).stem
# df.to_csv(eval_dir.joinpath(Path(f"{model_name}_summary.csv")), index=False)
# # Display the DataFrame
# print(df)

# # %%
# print(f"SUMMARY FOR RELU MODEL")
# model_1.print_summary()

# print("\n\n\n")
# print(f"SUMMARY FOR SOFTMAX MODEL")
# model_2.print_summary()
# %%
# fig = baseline.get_summary_figure("global_bin_dsc", horizontal=True)
# out_figure = str(f"{eval_dir}/exp_1_baseline_global_bin_dsc_figure.png")

# fig.write_image(out_figure)

# %%
#metric = "global_bin_dsc"
#metric = "rq"
#metric = "sq_dsc"
#metric = "sq_assd"
metric = "pq_dsc"
suffix = "_postprocessing_comp_baseline"     # start with _

fig2 = make_curve_over_setups(
    {
        # "mse_0.125": mse_low_sig,
        # "mse_0.125_proc": mse_low_sig_processed,
        # "mse_0.5": mse_high_sig,
        # "mse_0.5_proc": mse_high_sig_processed,
        # "adw_0.125": adw_low_sig,
        # "adw_0.125_proc": adw_low_sig_processed,
        # "adw_0.5": adw_high_sig,
        # "adw_0.5_proc": adw_high_sig_processed,
        "baseline": baseline,
        "baseline_proc": baseline_processed,
    },
    groups=("et (3)","tc (1, 3)", "wt (1, 2, 3)"),
    metric=metric,
)

out_figure = str(f"{output_dir}/{metric}_brats_regions{suffix}.png")
fig2.savefig(out_figure)
# %%
