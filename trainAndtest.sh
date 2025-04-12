#export CUBLAS_WORKSPACE_CONFIG=:4096:8
#mode option: train or test
#python train.py --title 'FlickCD_SYSU'\
#                --data_name 'SYSU'\
#                --mode 'train'\
#                --train_dataset_path  '../Dataset/SYSU/train/'\
#                --train_list_path '../Dataset/SYSU/train.txt'\
#                --val_dataset_path '../Dataset/SYSU/val/'\
#                --val_list_path '../Dataset/SYSU/val.txt'\
#                --test_dataset_path '../Dataset/SYSU/test/'\
#                --test_list_path '../Dataset/SYSU/test.txt'\
#                --learning_rate 5e-4\
#                --epochs 100\

#python train.py --title 'FlickCD_WHU'\
#                --data_name 'WHU'\
#                --mode 'train'\
#                --train_dataset_path  '../Dataset/WHU-CD/train/'\
#                --train_list_path '../Dataset/WHU-CD/train.txt'\
#                --val_dataset_path '../Dataset/WHU-CD/val/'\
#                --val_list_path '../Dataset/WHU-CD/val.txt'\
#                --test_dataset_path '../Dataset/WHU-CD/test/'\
#                --test_list_path '../Dataset/WHU-CD/test.txt'\
#                --learning_rate 2e-4\
#                --epochs 100\

#python train.py --title 'FlickCD_LEVIR-CD+'\
#                --data_name 'LEVIR+'\
#                --mode 'train'\
#                --train_dataset_path  '../Dataset/LEVIR-CD+/train/'\
#                --train_list_path '../Dataset/LEVIR-CD+/train.txt'\
#                --val_dataset_path '../Dataset/LEVIR-CD+/val/'\
#                --val_list_path '../Dataset/LEVIR-CD+/val.txt'\
#                --test_dataset_path '../Dataset/LEVIR-CD+/test/'\
#                --test_list_path '../Dataset/LEVIR-CD+/test.txt'\
#                --learning_rate 5e-4\
#                --epochs 200\

#python train.py --title 'FlickCD_CDD'\
#                --data_name 'CDD'\
#                --mode 'train'\
#                --train_dataset_path  '../Dataset/CDD/train/'\
#                --train_list_path '../Dataset/CDD/train.txt'\
#                --val_dataset_path '../Dataset/CDD/val/'\
#                --val_list_path '../Dataset/CDD/val.txt'\
#                --test_dataset_path '../Dataset/CDD/test/'\
#                --test_list_path '../Dataset/CDD/test.txt'\
#                --learning_rate 5e-4\
#                --epochs 250\

