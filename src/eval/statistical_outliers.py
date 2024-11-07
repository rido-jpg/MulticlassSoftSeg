import pandas as pd
from pathlib import Path

tsv_dir = Path("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/model_statistics/3D_UNet_v2__exp1_mse_softmax.tsv")
tsv_stem = tsv_dir.stem
save_path = tsv_dir.with_name(f"{tsv_stem}_outliers.csv")

# Load the dataset
data_df = pd.read_csv(tsv_dir, sep='\t')

def find_best_worst_subjects(df, metrics):
    results = {}
    for metric in metrics:
        best_subject = df.loc[df[metric].idxmax(), 'subject_name']
        worst_subject = df.loc[df[metric].idxmin(), 'subject_name']
        best_value = df[metric].max()
        worst_value = df[metric].min()
        results[metric] = {
            'Best Subject': best_subject,
            'Best Value': best_value,
            'Worst Subject': worst_subject,
            'Worst Value': worst_value
        }
    return pd.DataFrame(results).T

# Example usage
metrics_to_analyze = ['et (3)-pq_dsc','wt (1, 2, 3)-pq_dsc','tc (1, 3)-pq_dsc', 'et (3)-global_bin_dsc','wt (1, 2, 3)-global_bin_dsc','tc (1, 3)-global_bin_dsc',]    # group name-metric
results_df = find_best_worst_subjects(data_df, metrics_to_analyze)

results_df.to_csv(save_path)
# Display results
print(f"{tsv_dir.stem}:")

print(results_df)

