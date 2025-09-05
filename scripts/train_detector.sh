#!/bin/bash

CUR_DIR=$(pwd)

cd V-DETR
python3 main.py \
    --config configs/best.yaml \
    --ngpus 8 \
    --dataset_root_dir "${CUR_DIR}/dataset/pcd_dataset_slice_20"