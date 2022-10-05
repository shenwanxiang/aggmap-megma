## Package
https://pypi.org/project/deepmg/


## Installation

```bash
conda create -n deepmg python=3.6
conda activate deepmg

pip install numpy
pip install matplotlib
pip install ConfigParser
pip install pandas
pip install sklearn
pip install tensorflow==1.14
pip install keras==2.3.0
pip install keras_sequential_ascii
pip install minisom
pip install pillow
pip install deepmg
```


## Datasets 

```bash
wget https://drive.google.com/drive/folders/1XD8SqPHyEsLeFrIhIiqG5HdM9LopDfEw
unzip ./metagenomics.zip
```


## Running

```bash

## Cirrhosis
python3 -m deepmg --data_name='./metagenomics/cirphy' --type_emb="fill" --model='rf_model' --colormap='gray' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255

python3 -m deepmg --data_name='./metagenomics/cirphy' --type_emb="fill" --model='rf_model' --colormap='jet' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255


## ibd
python3 -m deepmg --data_name='./metagenomics/ibdphy' --type_emb="fill" --model='rf_model' --colormap='gray' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255

python3 -m deepmg --data_name='./metagenomics/ibdphy' --type_emb="fill" --model='rf_model' --colormap='jet' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255



## t2d
python3 -m deepmg --data_name='./metagenomics/t2dphy' --type_emb="fill" --model='rf_model' --colormap='gray' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255

python3 -m deepmg --data_name='./metagenomics/t2dphy' --type_emb="fill" --model='rf_model' --colormap='jet' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255


## Obesty
python3 -m deepmg --data_name='./metagenomics/obephy' --type_emb="fill" --model='rf_model' --colormap='gray' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255

python3 -m deepmg --data_name='./metagenomics/obephy' --type_emb="fill" --model='rf_model' --colormap='jet' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255

## CRC
python3 -m deepmg --data_name='./metagenomics/colphy' --type_emb="fill" --model='rf_model' --colormap='gray' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255

python3 -m deepmg --data_name='./metagenomics/colphy' --type_emb="fill" --model='rf_model' --colormap='jet' --original_data_folder='./' --type_bin='spb' --parent_folder_img='images' --save_para='y' --recreate_img=1 --type_run='config' --preprocess_img='vgg16' --coeff=255

```
