# %%
from panoptica.panoptica_statistics import Panoptica_Statistic


hard = Panoptica_Statistic.from_file("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/3D_UNet_v0__Hard_Baseline.tsv")
soft = Panoptica_Statistic.from_file("/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/eval/3D_UNet_v1__exp_1_mixed_loss.tsv")

print("\n\n\n")
print(f"SUMMARY FOR HARD MODEL")
hard.print_summary()

print("\n\n\n")
print(f"SUMMARY FOR SOFT MODEL")
soft.print_summary()
# %%
