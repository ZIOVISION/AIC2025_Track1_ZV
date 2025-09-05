#!/bin/bash

CUR_DIR=$(pwd)

cd ReIdModel

python train.py \
    --data_root "${CUR_DIR}/dataset/obj_crop_pcd_dataset" \
    --output_dir "${CUR_DIR}/weights/reid"

cd "${CUR_DIR}"

echo "ReID model training completed. Weights saved to weights/reid/"