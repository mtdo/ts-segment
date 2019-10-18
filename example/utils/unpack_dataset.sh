#!/bin/bash

mkdir data
mkdir data/MobiActV2
python3 utils/download_dataset_from_s3.py
unrar e data/MobiActV2/MobiAct_Dataset_v2.0.rar data/MobiActV2
mkdir data/MobiActV2/sessions
mkdir data/MobiActV2/sessions/csv
mkdir data/MobiActV2/sessions/txt
mv data/MobiActV2/*.csv data/MobiActV2/sessions/csv
mv data/MobiActV2/*.txt data/MobiActV2/sessions/txt
mv data/MobiActV2/sessions/txt/Readme.txt data/MobiActV2/README.txt
python3 utils/create_frames.py
