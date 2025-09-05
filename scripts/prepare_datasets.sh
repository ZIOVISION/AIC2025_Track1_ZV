#!/bin/bash

python3 tools/data_download.py \
    --data-root dataset

python3 tools/generate_pcd_data.py \
    --data-root dataset/MTMC_Tracking_2025 \
    --out-dir dataset/pcd_dataset

python3 tools/prepare_detector_train_data.py \
    --pcd-data-root dataset/pcd_dataset \
    --out-dir dataset/pcd_dataset_slice_20

python3 tools/prepare_reid_model_train_data.py \
    --pcd-data-root dataset/pcd_dataset \
    --out-dir dataset/obj_crop_pcd_dataset