import glob
import shutil
import os

dataset_directory = '../../../Dataset'
dataset_name = 'PillarDataset_Seg'
train_directory = 'train_val/Pillar_train'
val_directory = 'train_val/Pillar_val'
test_directory = 'test/Pillar_test'

output_name = 'PillarDataset_Seg_sf_xl'

multiplier = 100

os.makedirs(f'{dataset_directory}/{output_name}/train')

train_header = f'{dataset_directory}/{dataset_name}/{train_directory}'
train_path_list = glob.glob(f'{train_header}/**/**/*.png', recursive=True)
train_path_parsed = [p.split('\\')[-1].split('_') for p in train_path_list]
train_image_names = [f'@{float(m[0]) * multiplier}@{float(m[1]) * multiplier}@@@@@@@{m[3]}@{m[5]}@{m[7]}@@@@.png' for m in train_path_parsed]

for i in range(len(train_path_list)):
    shutil.copy(train_path_list[i], f'{dataset_directory}/{output_name}/train/{train_image_names[i]}')
    print(f'{i} / {len(train_path_list)}')



val_header = f'{dataset_directory}/{dataset_name}/{val_directory}'

os.makedirs(f'{dataset_directory}/{output_name}/val/database')
val_database_path_list = glob.glob(f'{val_header}/database/**/*.png', recursive=True)
val_database_path_parsed = [p.split('\\')[-1].split('_') for p in val_database_path_list]
val_database_image_names = [f'@{float(m[0]) * multiplier}@{float(m[1]) * multiplier}@@@@@@@{m[3]}@{m[5]}@{m[7]}@@@@.png' for m in val_database_path_parsed]

for i in range(len(val_database_path_list)):
    shutil.copy(val_database_path_list[i], f'{dataset_directory}/{output_name}/val/database/{val_database_image_names[i]}')
    print(f'{i} / {len(val_database_path_list)}')

os.makedirs(f'{dataset_directory}/{output_name}/val/queries')
val_query_path_list = glob.glob(f'{val_header}/query/**/*.png', recursive=True)
val_query_path_parsed = [p.split('\\')[-1].split('_') for p in val_query_path_list]
val_query_image_names = [f'@{float(m[0]) * multiplier}@{float(m[1]) * multiplier}@@@@@@@{m[3]}@{m[5]}@{m[7]}@@@@.png' for m in val_query_path_parsed]

for i in range(len(val_query_path_list)):
    shutil.copy(val_database_path_list[i], f'{dataset_directory}/{output_name}/val/queries/{val_query_image_names[i]}')
    print(f'{i} / {len(val_query_path_list)}')




test_header = f'{dataset_directory}/{dataset_name}/{test_directory}'

os.makedirs(f'{dataset_directory}/{output_name}/test/database')
test_database_path_list = glob.glob(f'{test_header}/database/**/*.png', recursive=True)
test_database_path_parsed = [p.split('\\')[-1].split('_') for p in test_database_path_list]
test_database_image_names = [f'@{float(m[0]) * multiplier}@{float(m[1]) * multiplier}@@@@@@@{m[3]}@{m[5]}@{m[7]}@@@@.png' for m in test_database_path_parsed]

for i in range(len(test_database_path_list)):
    shutil.copy(test_database_path_list[i], f'{dataset_directory}/{output_name}/test/database/{test_database_image_names[i]}')
    print(f'{i} / {len(test_database_path_list)}')

os.makedirs(f'{dataset_directory}/{output_name}/test/queries_v1')
test_query_path_list = glob.glob(f'{test_header}/query/**/*.png', recursive=True)
test_query_path_parsed = [p.split('\\')[-1].split('_') for p in test_query_path_list]
test_query_image_names = [f'@{float(m[0]) * multiplier}@{float(m[1]) * multiplier}@@@@@@@{m[3]}@{m[5]}@{m[7]}@@@@.png' for m in test_query_path_parsed]

for i in range(len(test_query_path_list)):
    shutil.copy(test_query_path_list[i], f'{dataset_directory}/{output_name}/test/queries_v1/{test_query_image_names[i]}')
    print(f'{i} / {len(test_query_path_list)}')



