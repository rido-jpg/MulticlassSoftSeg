from bids_dataset import mv_files_to_folders, split_data_and_save_to_csv, create_bids_path_list_of_dicts

split_csv_save_dir = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/data'
bids_root_path = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge/ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
split_target_path = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/data/external/ASNR-MICCAI-BraTS2023-GLI-Challenge'

list_of_path_dicts = create_bids_path_list_of_dicts(bids_root_path)
split_data_and_save_to_csv(list_of_path_dicts, [0.8, 0.1, 0.1], split_csv_save_dir, 42) 

split_csv = '/home/student/farid_ma/dev/multiclass_softseg/MulticlassSoftSeg/src/data/train_val_test_split.csv'

mv_files_to_folders(split_csv, bids_root_path,split_target_path)