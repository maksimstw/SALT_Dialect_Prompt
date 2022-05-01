run dialogpt
python dialogpt.py --file_path runs/21/dataset/dialogpt/all/cornell_movie_dataset_all.csv --output_path runs/21/gen/dialogpt/all/cornell_movie_dataset_all_gen.csv --cuda_id 5
python dialogpt.py --file_path runs/21/dataset/dialogpt/all/personachat_dataset_all.csv --output_path runs/21/gen/dialogpt/all/personachat_dataset_all_gen.csv --cuda_id 4
python dialogpt.py --file_path runs/21/dataset/dialogpt/all/dailydialog_dataset_all.csv --output_path runs/21/gen/dialogpt/all/dailydialog_dataset_all_gen.csv --cuda_id 3

run bst
python bst.py --file_path runs/21/dataset/bst/all/cornell_movie_dataset_all.csv --output_path runs/21/gen/bst/all/cornell_movie_dataset_all_gen.csv --cuda_id 2
python bst.py --file_path runs/21/dataset/bst/all/personachat_dataset_all.csv --output_path runs/21/gen/bst/all/personachat_dataset_all_gen.csv --cuda_id 1
python bst.py --file_path runs/21/dataset/bst/all/dailydialog_dataset_all.csv --output_path runs/21/gen/bst/all/dailydialog_dataset_all_gen.csv --cuda_id 0