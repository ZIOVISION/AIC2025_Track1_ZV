#!/bin/bash
CUR_DIR=$(pwd)

cd V-DETR
python3 inference.py \
    --config configs/best.yaml \
    --test_ckpt checkpoints/best/checkpoint_best.pth \
    --output_dir "${CUR_DIR}/outputs/detection" \
    --data_root "${CUR_DIR}/dataset/pcd_dataset" \
    --split_name "test" # "test" or "val"
