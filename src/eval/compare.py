# %%
from panoptica.panoptica_statistics import Panoptica_Statistic, make_curve_over_setups
from pathlib import Path
import pandas as pd

eval_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics")

baseline_path = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/3D_UNet_v0__Hard_Baseline.tsv"
path_1 = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/3D_UNet_v2__exp1_mse_relu.tsv"
path_2 = "/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/3D_UNet_v2__exp1_mse_softmax.tsv"

baseline = Panoptica_Statistic.from_file(baseline_path)
model_1 = Panoptica_Statistic.from_file(path_1)
model_2 = Panoptica_Statistic.from_file(path_2)

# %% 
data = model_2.get_summary_dict()

# Transform the data into a format suitable for CSV
rows = []
for category, metrics in data.items():
    for metric, (avg, std) in metrics.items():
        rows.append({
            'Category': category,
            'Metric': metric,
            'Average': avg,
            'StdDev': std
        })

# Create a DataFrame and save to CSV
df = pd.DataFrame(rows)
model_name = Path(path_2).stem
df.to_csv(eval_dir.joinpath(Path(f"{model_name}_summary.csv")), index=False)
# Display the DataFrame
print(df)

# %%
print(f"SUMMARY FOR RELU MODEL")
model_1.print_summary()

print("\n\n\n")
print(f"SUMMARY FOR SOFTMAX MODEL")
model_2.print_summary()
# %%
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