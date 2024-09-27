from panoptica.panoptica_statistics import Panoptica_Statistic


hard = Panoptica_Statistic.from_file("src/eval/3D_UNet_v1__baseline_softmax.tsv")
soft = Panoptica_Statistic.from_file("src/eval/3D_UNet_v7__MIXED_MSE.tsv")

print("\n\n\n")
print(f"SUMMARY FOR HARD MODEL")
hard.print_summary()

print("\n\n\n")
print(f"SUMMARY FOR SOFT MODEL")
soft.print_summary()